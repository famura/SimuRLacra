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

from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch as to
import torch.nn as nn

import pyrado
from pyrado.policies.base import Policy
from pyrado.policies.initialization import init_param
from pyrado.policies.recurrent.base import RecurrentPolicy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


class PotentialBasedPolicy(RecurrentPolicy, ABC):
    """Base class for policies that work with potential-based neutral networks"""

    name: str = None

    def __init__(
        self,
        spec: EnvSpec,
        obs_layer: [nn.Module, Policy],
        activation_nonlin: Callable,
        tau_init: float,
        tau_learnable: bool,
        kappa_init: float,
        kappa_learnable: bool,
        potential_init_learnable: bool,
        use_cuda: bool,
        hidden_size: Optional[int] = None,
    ):
        """
        Constructor

        :param spec: environment specification
        :param obs_layer: specify a custom PyTorch Module, by default (`None`) a linear layer with biases is used
        :param activation_nonlin: nonlinearity to compute the activations from the potential levels
        :param tau_init: initial value for the shared time constant of the potentials
        :param tau_learnable: flag to determine if the time constant is a learnable parameter or fixed
        :param kappa_init: initial value for the cubic decay, pass 0 (default) to disable cubic decay
        :param kappa_learnable: flag to determine if cubic decay is a learnable parameter or fixed
        :param potential_init_learnable: flag to determine if the initial potentials are a learnable parameter or fixed
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        :param hidden_size: number of neurons with potential, by default `None` which sets the number of hidden neurons
                            to the flat number of actions (in order to be compatible with ADNPolicy)
        """
        if not callable(activation_nonlin):
            raise pyrado.TypeErr(given=activation_nonlin, expected_type=Callable)

        super().__init__(spec, use_cuda)

        self._input_size = spec.obs_space.flat_dim  # observations include goal distance, prediction error, ect.
        self._hidden_size = spec.act_space.flat_dim if hidden_size is None else hidden_size
        self.num_recurrent_layers = 1
        self._potentials_max = 100.0  # clip potentials symmetrically at a very large value (for debugging)
        self._stimuli_external = to.zeros(self.hidden_size)
        self._stimuli_internal = to.zeros(self.hidden_size)

        # Create common layers
        self.resting_level = nn.Parameter(to.zeros(self.hidden_size), requires_grad=True)
        self.obs_layer = nn.Linear(self._input_size, self.hidden_size, bias=False) if obs_layer is None else obs_layer

        # Initial potential values
        self.potential_init_learnable = potential_init_learnable
        if potential_init_learnable:
            self._potentials_init = nn.Parameter(to.randn(self.hidden_size), requires_grad=True)
        else:
            if activation_nonlin is to.sigmoid:
                self._potentials_init = -7.0 * to.ones(self.hidden_size)
            else:
                self._potentials_init = to.zeros(self.hidden_size)

        # Potential dynamics's time constant
        self.tau_learnable = tau_learnable
        self._log_tau_init = to.log(to.tensor([tau_init], dtype=to.get_default_dtype()))
        if self.tau_learnable:
            self._log_tau = nn.Parameter(self._log_tau_init, requires_grad=True)
        else:
            self._log_tau = self._log_tau_init

        # Potential dynamics's cubic decay
        self.kappa_learnable = kappa_learnable
        if self.kappa_learnable or kappa_init != 0.0:
            self._log_kappa_init = to.log(to.tensor([kappa_init], dtype=to.get_default_dtype()))
            if self.kappa_learnable:
                self._log_kappa = nn.Parameter(self._log_kappa_init, requires_grad=True)
            else:
                self._log_kappa = self._log_kappa_init
        else:
            # Disable cubic decay
            self._log_kappa = to.tensor([1.0])

    def extra_repr(self) -> str:
        return (
            f"tau_learnable={self.tau_learnable}, kappa_learnable={self.kappa_learnable}, learn_init_potentials="
            f"{isinstance(self._potentials_init, nn.Parameter)}"
        )

    @property
    def hidden_size(self) -> int:
        return self.num_recurrent_layers * self._hidden_size

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
        """Get the time scale parameter."""
        return to.exp(self._log_tau)

    @property
    def kappa(self) -> to.Tensor:
        """Get the cubic decay parameter."""
        return to.exp(self._log_kappa)

    @abstractmethod
    def potentials_dot(self, potentials: to.Tensor, stimuli: to.Tensor) -> to.Tensor:
        """
        Compute the derivative of the neurons' potentials per time step.

        :param potentials: current potential values
        :param stimuli: sum of external and internal stimuli at the current point in time
        :return: time derivative of the potentials
        """
        raise NotImplementedError

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize common layers
            init_param(self.obs_layer, **kwargs)
            self.resting_level.data = to.randn_like(self.resting_level.data)

            # Initialize time constant and potentials if learnable
            if self.tau_learnable:
                self._log_tau.data = self._log_tau_init
            if self.potential_init_learnable:
                self._potentials_init.data = to.randn(self.hidden_size, device=self.device)

        else:
            self.param_values = init_values

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        if batch_size is None:
            return self._potentials_init.detach()  # needs to be detached for torch.jit.script()
        else:
            return self._potentials_init.detach().repeat(batch_size, 1)  # needs to be detached for torch.jit.script()

    def _unpack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Unpack the flat hidden state vector into a form the actual network module can use.
        Since hidden usually comes from some outer source, this method should validate it's shape.

        :param hidden: flat hidden state
        :param batch_size: if not `None`, hidden is 2-dim and the first dim represents parts of a data batch
        :return: unpacked hidden state of shape batch_size x channels_in x length_in, ready for the `Conv1d` module
        """
        if len(hidden.shape) == 1:
            assert (
                hidden.shape[0] == self.num_recurrent_layers * self._hidden_size
            ), "Passed hidden variable's size doesn't match the one required by the network."
            assert batch_size is None, "Cannot use batched observations with unbatched hidden state"
            return hidden.view(self.num_recurrent_layers * self._hidden_size)

        elif len(hidden.shape) == 2:
            assert (
                hidden.shape[1] == self.num_recurrent_layers * self._hidden_size
            ), "Passed hidden variable's size doesn't match the one required by the network."
            assert (
                hidden.shape[0] == batch_size
            ), f"Batch size of hidden state ({hidden.shape[0]}) must match batch size of observations ({batch_size})"
            return hidden.view(batch_size, self.num_recurrent_layers * self._hidden_size)

        else:
            raise RuntimeError(
                f"Improper shape of 'hidden'. Policy received {hidden.shape}, " f"but shape should be 1- or 2-dim"
            )

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
            return hidden.view(self.num_recurrent_layers * self._hidden_size)
        else:
            # Make sure that the batch dimension is the first element
            return hidden.view(batch_size, self.num_recurrent_layers * self._hidden_size)

    @abstractmethod
    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        raise NotImplementedError

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = "hidden_states") -> to.Tensor:
        if not rollout.data_format == "torch":
            raise pyrado.TypeErr(msg="The rollout data passed to evaluate() must be of type torch.Tensor!")
        if not rollout.continuous:
            raise pyrado.ValueErr(msg="The rollout data passed to evaluate() from a continuous rollout!")

        # Set policy, i.e. PyTorch nn.Module, to evaluation mode
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

        # Set policy, i.e. PyTorch nn.Module, back to training mode
        self.train()

        return to.stack(act_list)
