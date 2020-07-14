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
from abc import ABC, abstractmethod
from torch.jit import ScriptModule, export, trace_module, script

from pyrado.policies.base import Policy
from pyrado.sampling.step_sequence import StepSequence


class RecurrentPolicy(Policy, ABC):
    """
    Base class for recurrent policies.
    The policy does not store the hidden state on it's own, so it requires two arguments: (observation, hidden) and
    returns two values: (action, new_hidden).
    The hidden tensor is an 1-dim vector of state variables with unspecified meaning. In the batching case,
    it should be a 2-dim array, where the first dimension is the batch size matching that of the observations.
    """

    @property
    def is_recurrent(self) -> bool:
        return True

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """ Get the number of hidden state variables. """
        raise NotImplementedError

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        """
        Provide initial values for the hidden parameters. This should usually be a zero tensor.

        :param batch_size: number of states to track in parallel
        :return: Tensor of batch_size x hidden_size
        """
        if batch_size is None:
            return to.zeros(self.hidden_size)
        else:
            return to.zeros(batch_size, self.hidden_size)

    @abstractmethod
    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        """
        :param obs: observation from the environment

        :param hidden: the network's hidden state. If None, use init_hidden()
        :return: action to be taken and new hidden state
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> to.Tensor:
        """
        Re-evaluate the given rollout and return a derivable action tensor.
        This method makes sure that the gradient is propagated through the hidden state.

        :param rollout: complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks.
                                   Change this string for value functions.
        :return: actions with gradient data
        """
        raise NotImplementedError

    def trace(self) -> ScriptModule:
        """
        Create a ScriptModule from this policy.
        The returned module will always have the signature `action = tm(observation, hidden)`.
        For recurrent networks, it returns a stateful module that keeps the hidden states internally.
        Such modules have a `reset()` method to reset the hidden states.
        """
        return script(StatefulRecurrentNetwork(self))


class StatefulRecurrentNetwork(nn.Module):
    """
    A scripted wrapper for a recurrent neural network that stores the hidden state.

    .. note::
        Use this for transfer to C++.
    """

    # Attributes
    input_size: int
    output_size: int

    def __init__(self, net: RecurrentPolicy):
        """
        Constructor

        :param net: non-recurrent network to wrap

        .. note::
            Must not be a script module
        """
        super().__init__()

        # Setup attributes
        self.input_size = net.env_spec.obs_space.flat_dim
        self.output_size = net.env_spec.act_space.flat_dim

        # Setup hidden state buffer
        hidden = net.init_hidden()
        self.register_buffer('hidden', hidden)

        # Trace network (using random observation and init hidden state)
        inputs = {
            'forward': (
                to.from_numpy(net.env_spec.obs_space.sample_uniform()).to(to.get_default_dtype()),
                hidden
            ),
            'init_hidden': tuple()
        }
        self.net = trace_module(net, inputs)

    @export
    def reset(self):
        """ Reset the hidden states. """
        # Zero out hidden state
        self.hidden.copy_(self.net.init_hidden())

    def forward(self, inp):
        # Run through network
        out, hid = self.net(inp, self.hidden)

        # Store new hidden state
        self.hidden.copy_(hid)

        return out
