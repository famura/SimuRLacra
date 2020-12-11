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
from typing import Callable, Sequence

import pyrado
from pyrado.policies.base import Policy
from pyrado.policies.initialization import init_param
from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.nn_layers import IndiNonlinLayer


def pd_linear(p: to.Tensor, s: to.Tensor, h: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Basic proportional dynamics

    $\tau \dot{p} = s - p$

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param h: resting level, a.k.a. constant offset
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint="0")
    return (s + h - p) / tau


def pd_cubic(p: to.Tensor, s: to.Tensor, h: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Basic proportional dynamics with additional cubic decay

    $\tau \dot{p} = s + h - p + \kappa (h - p)^3$

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param h: resting level, a.k.a. constant offset
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint="0")
    if not all(kwargs["kappa"] >= 0):
        raise pyrado.ValueErr(given=kwargs["kappa"], ge_constraint="0")
    return (s + h - p + kwargs["kappa"] * to.pow(h - p, 3)) / tau


def pd_capacity_21(p: to.Tensor, s: to.Tensor, h: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 2 stable ($p=-C$, $p=C$) and 1 unstable fix points ($p=0$) for $s=0$

    $\tau \dot{p} =  s - (h - p) (1 - \frac{(h - p)^2}{C^2})$

    .. note::
        Intended to be used with sigmoid activation function, e.g. for the position tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param h: resting level, a.k.a. constant offset
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint="0")
    return (s - (h - p) * (to.ones_like(p) - (h - p) ** 2 / kwargs["capacity"] ** 2)) / tau


def pd_capacity_21_abs(p: to.Tensor, s: to.Tensor, h: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 2 stable ($p=-C$, $p=C$) and 1 unstable fix points ($p=0$) for $s=0$

    $\tau \dot{p} =  s - (h - p) (1 - \frac{\left| h - p \right|}{C})$

    The "absolute version" of `pd_capacity_21` has a lower magnitude and a lower oder of the resulting polynomial.

    .. note::
        Intended to be used with sigmoid activation function, e.g. for the position tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param h: resting level, a.k.a. constant offset
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint="0")
    return (s - (h - p) * (to.ones_like(p) - to.abs(h - p) / kwargs["capacity"])) / tau


def pd_capacity_32(p: to.Tensor, s: to.Tensor, h: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 3 stable ($p=-C$, $p=0$, $p=C$) and 2 unstable fix points ($p=-C/2$, $p=C/2$) for $s=0$

    $\tau \dot{p} =  s - (h - p) (1 - \frac{(h - p)^2}{C^2}) (1 - \frac{(2(h - p))^2}{C^2})$

    .. note::
        Intended to be used with tanh activation function, e.g. for the velocity tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param h: resting level, a.k.a. constant offset
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint="0")
    return (
        s
        + (h - p)
        * (to.ones_like(p) - (h - p) ** 2 / kwargs["capacity"] ** 2)
        * (to.ones_like(p) - ((2 * (h - p)) ** 2 / kwargs["capacity"] ** 2))
    ) / tau


def pd_capacity_32_abs(p: to.Tensor, s: to.Tensor, h: to.Tensor, tau: to.Tensor, **kwargs) -> to.Tensor:
    r"""
    Capacity-based dynamics with 3 stable ($p=-C$, $p=0$, $p=C$) and 2 unstable fix points ($p=-C/2$, $p=C/2$) for $s=0$

    $\tau \dot{p} =  \left( s + (h - p) (1 - \frac{\left| (h - p) \right|}{C})
    (1 - \frac{2 \left| (h - p) \right|}{C}) \right)$

    The "absolute version" of `pd_capacity_32` is less skewed due to a lower oder of the resulting polynomial.

    .. note::
        Intended to be used with tanh activation function, e.g. for the velocity tasks in RcsPySim.

    :param p: potential, higher values lead to higher activations
    :param s: stimulus, higher values lead to larger changes of the potentials (depends on the dynamics function)
    :param h: resting level, a.k.a. constant offset
    :param tau: time scaling factor, higher values lead to slower changes of the potentials (linear dependency)
    :param kwargs: additional parameters to the potential dynamics
    """
    if not all(tau > 0):
        raise pyrado.ValueErr(given=tau, g_constraint="0")
    return (
        s
        + (h - p)
        * (to.ones_like(p) - to.abs(h - p) / kwargs["capacity"])
        * (to.ones_like(p) - 2 * to.abs(h - p) / kwargs["capacity"])
    ) / tau


class ADNPolicy(PotentialBasedPolicy):
    """
    Activation Dynamic Network (ADN)

    .. seealso::
        [1] T. Luksch, M. Gineger, M. Mühlig, T. Yoshiike, "Adaptive Movement Sequences and Predictive Decisions based
        on Hierarchical Dynamical Systems", IROS, 2012
    """

    name: str = "adn"

    def __init__(
        self,
        spec: EnvSpec,
        activation_nonlin: [Callable, Sequence[Callable]],
        potentials_dyn_fcn: Callable,
        obs_layer: [nn.Module, Policy] = None,
        tau_init: float = 10.0,
        tau_learnable: bool = True,
        kappa_init: float = 1e-3,
        kappa_learnable: bool = True,
        capacity_learnable: bool = True,
        potential_init_learnable: bool = False,
        init_param_kwargs: dict = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param activation_nonlin: nonlinearity for output layer, highly suggested functions:
                                  `to.sigmoid` for position `to.tasks`, tanh for velocity tasks
        :param potentials_dyn_fcn: function to compute the derivative of the neurons' potentials
        :param obs_layer: specify a custom Pytorch Module;
                          by default (`None`) a linear layer with biases is used
        :param tau_init: initial value for the shared time constant of the potentials
        :param tau_learnable: flag to determine if the time constant is a learnable parameter or fixed
        :param kappa_init: initial value for the cubic decay
        :param kappa_learnable: flag to determine if cubic decay is a learnable parameter or fixed
        :param capacity_learnable: flag to determine if capacity is a learnable parameter or fixed
        :param potential_init_learnable: flag to determine if the initial potentials are a learnable parameter or fixed
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
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
        )

        # Create custom ADNPolicy layers
        self.prev_act_layer = nn.Linear(self._hidden_size, self._hidden_size, bias=False)
        self.pot_to_act_layer = IndiNonlinLayer(
            self._hidden_size, nonlin=activation_nonlin, bias=False, weight=True
        )  # scaling weight equals beta in eq (4) of [1]

        # Potential dynamics
        self.potentials_dot_fcn = potentials_dyn_fcn
        self.capacity_learnable = capacity_learnable
        if potentials_dyn_fcn in [pd_capacity_21, pd_capacity_21_abs, pd_capacity_32, pd_capacity_32_abs]:
            if activation_nonlin is to.sigmoid:
                # sigmoid(7.) approx 0.999
                self._log_capacity_init = to.log(to.tensor([7.0], dtype=to.get_default_dtype()))
                self._log_capacity = (
                    nn.Parameter(self._log_capacity_init, requires_grad=True)
                    if self.capacity_learnable
                    else self._log_capacity_init
                )
            elif activation_nonlin is to.tanh:
                # tanh(3.8) approx 0.999
                self._log_capacity_init = to.log(to.tensor([3.8], dtype=to.get_default_dtype()))
                self._log_capacity = (
                    nn.Parameter(self._log_capacity_init, requires_grad=True)
                    if self.capacity_learnable
                    else self._log_capacity_init
                )
            else:
                raise pyrado.TypeErr(
                    msg="Only output nonlinearities of type torch.sigmoid and torch.tanh are supported"
                    "for capacity-based potential dynamics."
                )
        else:
            self._log_capacity = None

        # Initialize policy parameters
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def extra_repr(self) -> str:
        return super().extra_repr().join(f", capacity_learnable={self.capacity_learnable}")

    @property
    def capacity(self) -> [None, to.Tensor]:
        """ Get the capacity parameter (exists for capacity-based dynamics functions), else return `None`. """
        return None if self._log_capacity is None else to.exp(self._log_capacity)

    def potentials_dot(self, potentials: to.Tensor, stimuli: to.Tensor) -> to.Tensor:
        """
        Compute the derivative of the neurons' potentials per time step.
        $/tau /dot{u} = f(u, s, h)$

        :param potentials: current potential values
        :param stimuli: sum of external and internal stimuli at the current point in time
        :return: time derivative of the potentials
        """
        return self.potentials_dot_fcn(
            potentials, stimuli, self.resting_level, self.tau, kappa=self.kappa, capacity=self.capacity
        )

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        super().init_param(init_values, **kwargs)

        if init_values is None:
            # Initialize layers
            init_param(self.prev_act_layer, **kwargs)
            if kwargs.get("sigmoid_nlin", False):
                self.prev_act_layer.weight.data.fill_(-0.5)  # inhibit others
                for i in range(self.prev_act_layer.weight.data.shape[0]):
                    self.prev_act_layer.weight.data[i, i] = 1.0  # excite self
            init_param(self.pot_to_act_layer, **kwargs)

            # Initialize cubic decay and capacity if learnable
            if self.potentials_dot_fcn == pd_cubic and self.kappa_learnable:
                self._log_kappa.data = self._log_kappa_init
            elif self.potentials_dot_fcn in [pd_capacity_21, pd_capacity_21_abs, pd_capacity_32, pd_capacity_32_abs]:
                if self.capacity_learnable:
                    self._log_capacity.data = self._log_capacity_init

        else:
            self.param_values = init_values

    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        obs = obs.to(self.device)

        # We assume flattened observations, if they are 2d, they're batched.
        if len(obs.shape) == 1:
            batch_size = None
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
        else:
            raise pyrado.ShapeErr(
                msg=f"Improper shape of 'obs'. Policy received {obs.shape}," f"but shape should be 1- or 2-dim"
            )

        # Unpack hidden tensor (i.e. the potentials of the last step) if specified, else initialize them
        pot = self._unpack_hidden(hidden, batch_size) if hidden is not None else self.init_hidden(batch_size)

        # Don't track the gradient through the potentials
        pot = pot.detach()

        # Scale the previous potentials, and pass them through a nonlinearity. Could also subtract a bias.
        act_prev = self.pot_to_act_layer(pot)

        # ----------------
        # Activation Logic
        # ----------------

        # Combine the current input and the hidden variables from the last step
        self._stimuli_external = self.obs_layer(obs)
        self._stimuli_internal = self.prev_act_layer(act_prev)

        # Potential dynamics forward integration
        pot = pot + self.potentials_dot(pot, self._stimuli_external + self._stimuli_internal)  # dt = 1

        # Clip the potentials
        pot = pot.clamp(min=-self._potentials_max, max=self._potentials_max)

        # Compute the actions (scale the potentials, subtract a bias, and pass them through a nonlinearity)
        act = self.pot_to_act_layer(pot)  # actions = activations

        # Pack hidden state
        hidden_out = self._pack_hidden(pot, batch_size)

        # Return the next action and store the last one as a hidden variable
        return act, hidden_out
