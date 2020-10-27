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

import torch as to
import torch.nn as nn
from typing import Callable

import pyrado
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy
from pyrado.policies.base_recurrent import RecurrentPolicy
from pyrado.policies.initialization import init_param
from pyrado.utils.input_output import print_cbt
from pyrado.utils.nn_layers import MirrConv1d, IndiNonlinLayer


class NFPolicy(RecurrentPolicy):
    """
    Neural Fields (NF)
    
    .. seealso::
        [1] S.-I. Amari "Dynamics of Pattern Formation in Lateral-Inhibition Type Neural Fields",
        Biological Cybernetics, 1977
    """

    name: str = 'nf'

    def __init__(self,
                 spec: EnvSpec,
                 dt: float,
                 hidden_size: int,
                 obs_layer: [nn.Module, Policy] = None,
                 activation_nonlin: Callable = to.sigmoid,
                 mirrored_conv_weights: bool = True,
                 conv_out_channels: int = 1,
                 conv_kernel_size: int = None,
                 conv_padding_mode: str = 'circular',
                 tau_init: float = 1.,
                 tau_learnable: bool = True,
                 kappa_init: float = None,
                 kappa_learnable: bool = True,
                 potential_init_learnable: bool = False,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param dt: time step size
        :param hidden_size: number of neurons with potential
        :param obs_layer: specify a custom PyTorch Module, by default (`None`) a linear layer with biases is used
        :param activation_nonlin: nonlinearity to compute the activations from the potential levels
        :param mirrored_conv_weights: re-use weights for the second half of the kernel to create a "symmetric" kernel
        :param conv_out_channels: number of filter for the 1-dim convolution along the potential-based neurons
        :param conv_kernel_size: size of the kernel for the 1-dim convolution along the potential-based neurons
        :param tau_init: initial value for the shared time constant of the potentials
        :param tau_learnable: flag to determine if the time constant is a learnable parameter or fixed
        :param kappa_init: initial value for the cubic decay, pass `None` (default) to disable cubic decay
        :param kappa_learnable: flag to determine if cubic decay is a learnable parameter or fixed
        :param potential_init_learnable: flag to determine if the initial potentials are a learnable parameter or fixed
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(dt, (float, int)):
            raise pyrado.TypeErr(given=dt, expected_type=float)
        if not isinstance(hidden_size, int):
            raise pyrado.TypeErr(given=hidden_size, expected_type=int)
        if hidden_size < 2:
            raise pyrado.ValueErr(given=hidden_size, g_constraint='1')
        if conv_kernel_size is None:
            conv_kernel_size = hidden_size
        if not conv_kernel_size%2 == 1:
            print_cbt(f'Increased the kernel size {conv_kernel_size} the next odd number ({conv_kernel_size + 1}) '
                      f'in order to obtain shape-conserving padding.', 'y')
            conv_kernel_size = conv_kernel_size + 1
        if conv_padding_mode not in ['circular', 'reflected', 'zeros']:
            raise pyrado.ValueErr(given=conv_padding_mode, eq_constraint='circular, reflected, or zeros')
        if not callable(activation_nonlin):
            raise pyrado.TypeErr(given=activation_nonlin, expected_type=Callable)

        super().__init__(spec, use_cuda)

        # Store inputs
        self._dt = to.tensor([dt], dtype=to.get_default_dtype())
        self._input_size = spec.obs_space.flat_dim  # observations include goal distance, prediction error, ect.
        self._hidden_size = hidden_size  # number of potential neurons
        self.num_recurrent_layers = 1
        self.mirrored_conv_weights = mirrored_conv_weights

        # Create the layers
        self.obs_layer = nn.Linear(self._input_size, self._hidden_size, bias=False) if obs_layer is None else obs_layer
        self.resting_level = nn.Parameter(to.zeros(hidden_size), requires_grad=True)
        padding = conv_kernel_size//2 if conv_padding_mode != 'circular' else conv_kernel_size - 1  # 1 means no padding
        conv1d_class = MirrConv1d if mirrored_conv_weights else nn.Conv1d
        self.conv_layer = conv1d_class(
            in_channels=1,  # treat potentials as a time series of values (convolutions is over the "time" axis)
            out_channels=conv_out_channels,
            kernel_size=conv_kernel_size, padding_mode=conv_padding_mode, padding=padding, bias=False,
            stride=1, dilation=1, groups=1  # defaults
        )
        # self.post_conv_layer = nn.Linear(conv_out_channels, spec.act_space.flat_dim, bias=False)
        self.pot_to_activ = IndiNonlinLayer(self._hidden_size, nonlin=activation_nonlin, bias=False, weight=True)
        self.act_layer = nn.Linear(self._hidden_size, spec.act_space.flat_dim, bias=False)

        # Call custom initialization function after PyTorch network parameter initialization
        self._potentials_max = 100.  # clip potentials symmetrically at a very large value (for debugging)
        self._stimuli_external = to.zeros(self.hidden_size)
        self._stimuli_internal = to.zeros(self.hidden_size)
        self.potential_init_learnable = potential_init_learnable
        if potential_init_learnable:
            self._potentials_init = nn.Parameter(to.randn(self.hidden_size), requires_grad=True)
        else:
            if activation_nonlin is to.sigmoid:
                self._potentials_init = -7.*to.ones(self.hidden_size)
            else:
                self._potentials_init = to.zeros(self.hidden_size)

        # Potential dynamics's time constant
        self.tau_learnable = tau_learnable
        self._log_tau_init = to.log(to.tensor([tau_init], dtype=to.get_default_dtype()))
        self._log_tau = nn.Parameter(self._log_tau_init, requires_grad=True) \
            if self.tau_learnable else self._log_tau_init

        if kappa_init is not None:
            self.kappa_learnable = kappa_learnable
            self._log_kappa_init = to.log(to.tensor([kappa_init], dtype=to.get_default_dtype()))
            self._log_kappa = nn.Parameter(self._log_kappa_init, requires_grad=True) \
                if self.kappa_learnable else self._log_kappa_init
        else:
            self._log_kappa = None

        # Initialize policy parameters
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def extra_repr(self) -> str:
        return f'tau_learnable={self.tau_learnable}, use_kappa={self._log_kappa is not None}, learn_init_potentials=' \
               f'{isinstance(self._potentials_init, nn.Parameter)}'

    @property
    def hidden_size(self) -> int:
        return self.num_recurrent_layers*self._hidden_size

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        if batch_size is None:
            return self._potentials_init.detach()  # needs to be detached for torch.jit.script()
        else:
            return self._potentials_init.detach().repeat(batch_size, 1)  # needs to be detached for torch.jit.script()

    @property
    def stimuli_external(self) -> to.Tensor:
        """
        Get the neurons' external stimuli, resulting from the current observations.
        This is used for recording during a rollout.
        """
        return self._stimuli_external

    @property
    def stimuli_internal(self) -> to.Tensor:
        """
        Get the neurons' internal stimuli, resulting from the previous activations of the neurons.
        This is used for recording during a rollout.
        """
        return self._stimuli_internal

    @property
    def tau(self) -> to.Tensor:
        """ Get the time scale parameter (exists for all potential dynamics functions). """
        return to.exp(self._log_tau)

    @property
    def kappa(self) -> [None, to.Tensor]:
        """ Get the cubic decay parameter if specified in the constructor, else return zero. """
        return None if self._log_kappa is None else to.exp(self._log_kappa)

    def potentials_dot(self, potentials: to.Tensor, stimuli: to.Tensor) -> to.Tensor:
        r"""
        Compute the derivative of the neurons' potentials per time step.
        $/tau /dot{u} = s + h - u + /kappa (h - u)^3,
        /quad /text{with} s = s_{int} + s_{ext} = W*o + /int{w(u, v) f(u) dv}$

        :param potentials: current potential values
        :param stimuli: sum of external and internal stimuli at the current point in time
        :return: time derivative of the potentials
        """
        rhs = stimuli + self.resting_level - potentials
        if self._log_kappa is not None:
            rhs += self.kappa*to.pow(self.resting_level - potentials, 3)
        return rhs/self.tau

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize RNN layers
            init_param(self.obs_layer, **kwargs)
            self.resting_level.data = to.randn_like(self.resting_level.data)
            init_param(self.conv_layer, **kwargs)
            # init_param(self.post_conv_layer, **kwargs)
            init_param(self.pot_to_activ, **kwargs)
            init_param(self.act_layer, **kwargs)

            # Initialize time constant if learnable
            if self.tau_learnable:
                self._log_tau.data = self._log_tau_init
            # Initialize the potentials if learnable
            if self.potential_init_learnable:
                self._potentials_init.data = to.randn(self.hidden_size)

        else:
            self.param_values = init_values

    def _unpack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Unpack the flat hidden state vector into a form the actual network module can use.
        Since hidden usually comes from some outer source, this method should validate it's shape.

        :param hidden: flat hidden state
        :param batch_size: if not `None`, hidden is 2-dim and the first dim represents parts of a data batch
        :return: unpacked hidden state of shape batch_size x channels_in x length_in, ready for the `Conv1d` module
        """
        if len(hidden.shape) == 1:
            assert hidden.shape[0] == self.num_recurrent_layers*self._hidden_size, \
                "Passed hidden variable's size doesn't match the one required by the network."
            assert batch_size is None, 'Cannot use batched observations with unbatched hidden state'
            return hidden.view(self.num_recurrent_layers*self._hidden_size)

        elif len(hidden.shape) == 2:
            assert hidden.shape[1] == self.num_recurrent_layers*self._hidden_size, \
                "Passed hidden variable's size doesn't match the one required by the network."
            assert hidden.shape[0] == batch_size, \
                f'Batch size of hidden state ({hidden.shape[0]}) must match batch size of observations ({batch_size})'
            return hidden.view(batch_size, self.num_recurrent_layers*self._hidden_size)

        else:
            raise RuntimeError(f"Improper shape of 'hidden'. Policy received {hidden.shape}, "
                               f"but shape should be 1- or 2-dim")

    def _pack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Pack the hidden state returned by the network into an 1-dim state vector.
        This should be the reverse operation of `_unpack_hidden`.

        :param hidden: hidden state as returned by the network
        :param batch_size: if not `None`, the result should be 2-dim and the first dim represents parts of a data batch
        :return: packed hidden state
        """
        if batch_size is None:
            # Simply flatten the hidden state
            return hidden.view(self.num_recurrent_layers*self._hidden_size)
        else:
            # Make sure that the batch dimension is the first element
            return hidden.view(batch_size, self.num_recurrent_layers*self._hidden_size)

    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        """
        Compute the goal distance, prediction error, and predicted cost.
        Then pass it to the wrapped RNN.

        :param obs: observations coming from the environment i.e. noisy
        :param hidden: current hidden states, in this case action and potentials of the last time step
        :return: current action and new hidden states
        """
        obs = obs.to(self.device)

        # We assume flattened observations, if they are 2d, they're batched.
        if len(obs.shape) == 1:
            batch_size = None
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
        else:
            raise pyrado.ShapeErr(msg=f"Improper shape of 'obs'. Policy received {obs.shape},"
                                      f"but shape should be 1- or 2-dim")

        # Unpack hidden tensor (i.e. the potentials of the last step) if specified, else initialize them
        pot = self._unpack_hidden(hidden, batch_size) if hidden is not None else self.init_hidden(batch_size)

        # Don't track the gradient through the potentials
        pot = pot.detach()

        # Scale the previous potentials, subtract a bias, and pass them through a nonlinearity
        activations_prev = self.pot_to_activ(pot)

        # ----------------
        # Activation Logic
        # ----------------

        # Combine the current inputs
        self._stimuli_external = self.obs_layer(obs)

        # Reshape and convolve
        b = batch_size if batch_size is not None else 1
        self._stimuli_internal = self.conv_layer(activations_prev.view(b, 1, self._hidden_size))
        self._stimuli_internal = to.sum(self._stimuli_internal,
                                        dim=1)  # TODO do multiple out channels makes sense if just summed up?
        self._stimuli_internal = self._stimuli_internal.squeeze()

        # Combine the different output channels of the convolution
        # stimulus_pot = self.post_conv_layer(stimulus_pot)

        # Check the shapes before adding the resting level since the broadcasting could mask errors from the convolution
        if not self._stimuli_external.shape == self._stimuli_internal.shape:
            raise pyrado.ShapeErr(given=self._stimuli_internal, expected_match=self._stimuli_external)

        # Potential dynamics forward integration
        pot = pot + self._dt*self.potentials_dot(pot, self._stimuli_external + self._stimuli_internal)

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

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> to.Tensor:
        self.eval()
        act_list = []

        for ro in rollout.iterate_rollouts():
            if hidden_states_name in rollout.data_names:
                # Get initial hidden state from first step
                hidden = ro[0][hidden_states_name]
            else:
                # Let the network pick the default hidden state
                hidden = None

            # Run steps consecutively reusing the hidden state
            for step in ro:
                act, hidden = self(step.observation, hidden)
                act_list.append(act)

        return to.stack(act_list)
