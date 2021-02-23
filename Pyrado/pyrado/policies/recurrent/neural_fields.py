# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import multiprocessing as mp
import torch as to
import torch.nn as nn
from typing import Callable

import pyrado
from pyrado.policies.base import Policy
from pyrado.policies.initialization import init_param
from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt
from pyrado.utils.nn_layers import MirrConv1d, IndiNonlinLayer


class NFPolicy(PotentialBasedPolicy):
    """
    Neural Fields (NF)

    .. seealso::
        [1] S.-I. Amari "Dynamics of Pattern Formation in Lateral-Inhibition Type Neural Fields",
        Biological Cybernetics, 1977
    """

    name: str = "nf"

    def __init__(
        self,
        spec: EnvSpec,
        hidden_size: int,
        obs_layer: [nn.Module, Policy] = None,
        activation_nonlin: Callable = to.sigmoid,
        mirrored_conv_weights: bool = True,
        conv_out_channels: int = 1,
        conv_kernel_size: int = None,
        conv_padding_mode: str = "circular",
        tau_init: float = 10.0,
        tau_learnable: bool = True,
        kappa_init: float = 0.0,
        kappa_learnable: bool = True,
        potential_init_learnable: bool = False,
        init_param_kwargs: dict = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param hidden_size: number of neurons with potential
        :param obs_layer: specify a custom PyTorch Module, by default (`None`) a linear layer with biases is used
        :param activation_nonlin: nonlinearity to compute the activations from the potential levels
        :param mirrored_conv_weights: re-use weights for the second half of the kernel to create a "symmetric" kernel
        :param conv_out_channels: number of filter for the 1-dim convolution along the potential-based neurons
        :param conv_kernel_size: size of the kernel for the 1-dim convolution along the potential-based neurons
        :param tau_init: initial value for the shared time constant of the potentials
        :param tau_learnable: flag to determine if the time constant is a learnable parameter or fixed
        :param kappa_init: initial value for the cubic decay, pass 0 (default) to disable cubic decay
        :param kappa_learnable: flag to determine if cubic decay is a learnable parameter or fixed
        :param potential_init_learnable: flag to determine if the initial potentials are a learnable parameter or fixed
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(hidden_size, int):
            raise pyrado.TypeErr(given=hidden_size, expected_type=int)
        if hidden_size < 2:
            raise pyrado.ValueErr(given=hidden_size, g_constraint="1")
        if conv_kernel_size is None:
            conv_kernel_size = hidden_size
        if not conv_kernel_size % 2 == 1:
            print_cbt(
                f"Increased the kernel size {conv_kernel_size} the next odd number ({conv_kernel_size + 1}) "
                f"in order to obtain shape-conserving padding.",
                "y",
            )
            conv_kernel_size = conv_kernel_size + 1
        if conv_padding_mode not in ["circular", "reflected", "zeros"]:
            raise pyrado.ValueErr(given=conv_padding_mode, eq_constraint="circular, reflected, or zeros")
        if not callable(activation_nonlin):
            raise pyrado.TypeErr(given=activation_nonlin, expected_type=Callable)

        # Set the multiprocessing start method to spawn, since PyTorch is using the GPU for convolutions if it can
        if to.cuda.is_available() and mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)

        super().__init__(
            spec,
            obs_layer,
            activation_nonlin,
            tau_init,
            tau_learnable,
            kappa_init,
            kappa_learnable,
            potential_init_learnable,
            use_cuda,
            hidden_size,
        )

        # Create custom NFPolicy layers
        self.mirrored_conv_weights = mirrored_conv_weights
        padding = (
            conv_kernel_size // 2 if conv_padding_mode != "circular" else conv_kernel_size - 1
        )  # 1 means no padding
        conv1d_class = MirrConv1d if mirrored_conv_weights else nn.Conv1d
        self.conv_layer = conv1d_class(
            in_channels=1,  # treat potentials as a time series of values (convolutions is over the "time" axis)
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size,
            padding_mode=conv_padding_mode,
            padding=padding,
            bias=False,
            stride=1,
            dilation=1,
            groups=1,  # defaults
        )
        # self.post_conv_layer = nn.Linear(conv_out_channels, spec.act_space.flat_dim, bias=False)
        self.pot_to_activ = IndiNonlinLayer(self.hidden_size, nonlin=activation_nonlin, bias=True, weight=True)
        self.act_layer = nn.Linear(self.hidden_size, spec.act_space.flat_dim, bias=False)

        # Initialize policy parameters
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def potentials_dot(self, potentials: to.Tensor, stimuli: to.Tensor) -> to.Tensor:
        r"""
        Compute the derivative of the neurons' potentials per time step.
        $/tau /dot{u} = s + h - u + /kappa (h - u)^3,
        /quad /text{with} s = s_{int} + s_{ext} = W*o + /int{w(u, v) f(u) dv}$

        :param potentials: current potential values
        :param stimuli: sum of external and internal stimuli at the current point in time
        :return: time derivative of the potentials
        """
        rhs = stimuli + self.resting_level - potentials + self.kappa * to.pow(self.resting_level - potentials, 3)
        return rhs / self.tau

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        super().init_param(init_values, **kwargs)

        if init_values is None:
            # Initialize layers
            init_param(self.conv_layer, **kwargs)
            # init_param(self.post_conv_layer, **kwargs)
            init_param(self.pot_to_activ, **kwargs)
            init_param(self.act_layer, **kwargs)

        else:
            self.param_values = init_values

    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        obs = obs.to(device=self.device, dtype=to.get_default_dtype())

        # We assume flattened observations, if they are 2d, they're batched.
        if len(obs.shape) == 1:
            batch_size = None
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
        else:
            raise pyrado.ShapeErr(msg=f"Expected 1- or 2-dim observations, but the shape is {obs.shape}!")

        # Unpack hidden tensor (i.e. the potentials of the last step) if specified, else initialize them
        if hidden is not None:
            hidden = hidden.to(device=self.device, dtype=to.get_default_dtype())
            pot = self._unpack_hidden(hidden, batch_size)
        else:
            pot = self.init_hidden(batch_size)

        # Don't track the gradient through the potentials
        pot = pot.detach()

        # Scale the previous potentials, subtract a bias, and pass them through a nonlinearity
        activations_prev = self.pot_to_activ(pot)

        # ----------------
        # Activation Logic
        # ----------------

        # Combine the current inputs
        self._stimuli_external = self.obs_layer(obs).squeeze()

        # Reshape and convolve
        b = batch_size if batch_size is not None else 1
        self._stimuli_internal = self.conv_layer(activations_prev.view(b, 1, self._hidden_size))
        self._stimuli_internal = to.sum(
            self._stimuli_internal, dim=1
        )  # TODO do multiple out channels makes sense if just summed up?
        self._stimuli_internal = self._stimuli_internal.squeeze()

        # Combine the different output channels of the convolution
        # stimulus_pot = self.post_conv_layer(stimulus_pot)

        # Check the shapes before adding the resting level since the broadcasting could mask errors from the convolution
        if not self._stimuli_external.shape == self._stimuli_internal.shape:
            raise pyrado.ShapeErr(given=self._stimuli_internal, expected_match=self._stimuli_external)

        # Potential dynamics forward integration
        pot = pot + self.potentials_dot(pot, self._stimuli_external + self._stimuli_internal)  # dt = 1

        # Clip the potentials
        pot = pot.clamp(min=-self._potentials_max, max=self._potentials_max)

        # Compute the activations (scale the potentials, subtract a bias, and pass them through a nonlinearity)
        activ = self.pot_to_activ(pot)

        # Compute the actions from the activations
        act = self.act_layer(activ)

        # Pack hidden state
        hidden_out = self._pack_hidden(pot, batch_size)

        # Return the next action and store the current potentials as a hidden variable
        return act, hidden_out
