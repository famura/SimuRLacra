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
import torch.nn.utils.convert_parameters as cp
from abc import ABC, abstractmethod
from torch.jit import ScriptModule, trace, script
from warnings import warn

import pyrado
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


def _get_or_create_grad(t):
    """
    Get the grad tensor for the given tensor, or create if missing.

    :param t: input tensor
    :return g: gradient attribute of input tensor, zeros if created
    """
    g = t.grad
    if g is None:
        g = to.zeros_like(t)
        t.grad = g
    return g


class Policy(nn.Module, ABC):
    """ Base class for all policies in Pyrado """

    name: str = None  # unique identifier

    def __init__(self, spec: EnvSpec, use_cuda: bool):
        """
        Constructor

        :param spec: environment specification
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(spec, EnvSpec):
            raise pyrado.TypeErr(given=spec, expected_type=EnvSpec)
        if not isinstance(use_cuda, bool):
            raise pyrado.TypeErr(given=use_cuda, expected_type=bool)

        # Call torch.nn.Module's constructor
        super().__init__()

        self._env_spec = spec
        if not use_cuda:
            self._device = "cpu"
        elif use_cuda and to.cuda.is_available():
            self._device = "cuda"
        elif use_cuda and not to.cuda.is_available():
            warn("Tried to run on CUDA, but it is not available. Falling back to CPU.")
            self._device = "cpu"

    @property
    def device(self) -> str:
        """ Get the device (CPU or GPU) on which the policy is stored. """
        return self._device

    @property
    def env_spec(self) -> EnvSpec:
        """ Get the specification of environment the policy acts in. """
        return self._env_spec

    @property
    def param_values(self) -> to.Tensor:
        """
        Get the parameters of the policy as 1d array.
        The values are copied, modifying the return value does not propagate to the actual policy parameters.
        However, setting this variable will change the parameters.
        """
        return cp.parameters_to_vector(self.parameters())

    @param_values.setter
    def param_values(self, param: to.Tensor):
        """ Set the policy parameters from an 1d array. """
        if not self.param_values.shape == param.shape:
            raise pyrado.ShapeErr(given=param, expected_match=self.param_values)
        cp.vector_to_parameters(param, self.parameters())

    @property
    def param_grad(self) -> to.Tensor:
        """
        Get the gradient of the parameters as 1d array.
        The values are copied, modifying the return value does not propagate to the actual policy parameters.
        However, setting this variable will change the gradient.
        """
        return cp.parameters_to_vector(_get_or_create_grad(p) for p in self.parameters())

    @param_grad.setter
    def param_grad(self, param):
        """ Set the policy parameter gradient from an 1d array. """
        cp.vector_to_parameters(param, (_get_or_create_grad(p) for p in self.parameters()))

    @property
    def num_param(self) -> int:
        """ Get the number of policy parameters. """
        return sum(p.data.numel() for p in self.parameters())

    @property
    def is_recurrent(self) -> bool:
        """ Bool to signalise it the policy has a recurrent architecture. """
        return False

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        """
        Provide initial values for the hidden parameters. This should usually be a zero tensor.
        The default implementation will raise an error, to enforce override this function for recurrent policies.

        :param batch_size: number of states to track in parallel
        :return: Tensor of batch_size x hidden_size
        """
        raise AttributeError(
            "Only recurrent policies should use the init_hidden() method."
            "Make sure to implement this function for every recurrent policy type."
        )

    @abstractmethod
    def init_param(self, init_values: to.Tensor = None, **kwargs):
        """
        Initialize the policy's parameters. By default the parameters are initialized randomly.

        :param init_values: tensor of fixed initial policy parameter values
        :param kwargs: additional keyword arguments for the policy parameter initialization
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the policy to it's initial state.
        This should be called at the start of a rollout. Stateful policies should use it to reset the state variables.
        The default implementation does nothing.
        """
        pass  # this is used in rollout() even though your IDE might not link it

    @abstractmethod
    def forward(self, *args, **kwargs) -> [to.Tensor, (to.Tensor, to.Tensor)]:
        """
        Get the action according to the policy and the observations (forward pass).

        :param args: inputs, e.g. an observation from the environment or an observation and a hidden state
        :param kwargs: inputs, e.g. an observation from the environment or an observation and a hidden state
        :return: outputs, e.g. an action or an action and a hidden state
        """
        raise NotImplementedError

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = "hidden_states") -> to.Tensor:
        """
        Re-evaluate the given rollout and return a derivable action tensor.
        The default implementation simply calls `forward()`.

        :param rollout: complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks.
                                   Defaults to 'hidden_states'. Change for value functions.
        :return: actions with gradient data
        """
        # Set policy, i.e. PyTorch nn.Module, to evaluation mode
        self.eval()

        res = self(rollout.get_data_values("observations", truncate_last=True))  # all observations at once

        # Set policy, i.e. PyTorch nn.Module, back to training mode
        self.train()

        return res

    def script(self) -> ScriptModule:
        """
        Create a ScriptModule from this policy.
        The returned module will always have the signature `action = tm(observation)`.
        For recurrent networks, it returns a stateful module that keeps the hidden states internally.
        Such modules have a reset() method to reset the hidden states.
        """
        # This does not work for recurrent policies, which is why they override this function.
        return script(TracedPolicyWrapper(self))


class TracedPolicyWrapper(nn.Module):
    """ Wrapper for a traced policy. Mainly used to add `input_size` and `output_size` attributes. """

    # Attributes
    input_size: int
    output_size: int

    def __init__(self, module: Policy):
        """
        Constructor

        :param module: non-recurrent network to wrap, which must not be a script module
        """
        super().__init__()

        # Setup attributes
        self.input_size = module.env_spec.obs_space.flat_dim
        self.output_size = module.env_spec.act_space.flat_dim

        self.module = trace(module, (to.from_numpy(module.env_spec.obs_space.sample_uniform()),))

    def forward(self, obs):
        return self.module(obs)


class TwoHeadedPolicy(Policy, ABC):
    """ Base class for policies with a shared body and two separate heads. """

    @abstractmethod
    def init_param(self, init_values: to.Tensor = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, obs: to.Tensor) -> [to.Tensor, (to.Tensor, to.Tensor)]:
        raise NotImplementedError
