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

import torch as to
import torch.nn as nn
from torch.jit import ScriptModule, export, script, trace_module

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
        """Get the number of hidden state variables."""
        raise NotImplementedError

    def init_hidden(self, batch_size: int = None) -> to.Tensor:
        """
        Provide initial values for the hidden parameters. This should usually be a zero tensor.

        :param batch_size: number of states to track in parallel
        :return: Tensor of batch_size x hidden_size
        """
        if batch_size is None:
            return to.zeros(self.hidden_size, device=self.device)
        else:
            return to.zeros(batch_size, self.hidden_size, device=self.device)

    @abstractmethod
    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
        """
        :param obs: observation from the environment

        :param hidden: the network's hidden state. If None, use init_hidden()
        :return: action to be taken and new hidden state
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, rollout: StepSequence, hidden_states_name: str = "hidden_states") -> to.Tensor:
        """
        Re-evaluate the given rollout and return a derivable action tensor.
        This method makes sure that the gradient is propagated through the hidden state.

        :param rollout: complete rollout
        :param hidden_states_name: name of hidden states rollout entry, used for recurrent networks.
                                   Change this string for value functions.
        :return: actions with gradient data
        """
        raise NotImplementedError

    def script(self) -> ScriptModule:
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
        # Make absolutely sure there are no leftover back-connections here!
        hidden_buf = to.zeros_like(hidden)
        hidden_buf.data.copy_(hidden.data)
        self.register_buffer("hidden", hidden_buf)

        # Trace network (using random observation and init hidden state)
        inputs = {
            "forward": (
                to.from_numpy(net.env_spec.obs_space.sample_uniform()).to(to.get_default_dtype()),
                hidden_buf.to(to.get_default_dtype()),
            ),
            "init_hidden": tuple(),  # call with no arguments
        }
        self.net = trace_module(net, inputs)

    @export
    def reset(self):
        """Reset the hidden states."""
        self.hidden.data.copy_(self.net.init_hidden().data)  # in most cases all zeros

    def forward(self, inp):
        # Run through network
        out, hid = self.net(inp, self.hidden)

        # Store new hidden state
        self.hidden.data.copy_(hid.data)

        return out


def default_unpack_hidden(hidden: to.Tensor, num_recurrent_layers: int, hidden_size: int, batch_size: int = None):
    """
    Unpack the flat hidden state vector into the form expected by torch.nn.RNNBase subclasses.

    :param hidden: packed hidden state
    :param num_recurrent_layers: number of recurrent layers
    :param hidden_size: size of the hidden layers (all equal)
    :param batch_size: if not none, hidden is 2d, and the first dimension represents parts of a data batch
    :return: unpacked hidden state, a tensor of num_recurrent_layers x batch_size x hidden_size.
    """
    if len(hidden.shape) == 1:
        assert (
            hidden.shape[0] == num_recurrent_layers * hidden_size
        ), "Passed hidden variable's size doesn't match the one required by the network."
        # we could handle that case, but for now it's not necessary.
        assert batch_size is None, "Cannot use batched observations with unbatched hidden state"
        return hidden.view(num_recurrent_layers, 1, hidden_size)

    elif len(hidden.shape) == 2:
        assert (
            hidden.shape[1] == num_recurrent_layers * hidden_size
        ), "Passed hidden variable's size doesn't match the one required by the network."
        assert (
            hidden.shape[0] == batch_size
        ), f"Batch size of hidden state ({hidden.shape[0]}) must match batch size of observations ({batch_size})"
        return hidden.view(batch_size, num_recurrent_layers, hidden_size).permute(1, 0, 2)

    else:
        raise RuntimeError(
            f"Improper shape of 'hidden'. Policy received {hidden.shape}, " f"but shape should be 1- or 2-dim"
        )


def default_pack_hidden(hidden: to.Tensor, num_recurrent_layers, hidden_size: int, batch_size: int = None):
    """
    Pack the hidden state returned by torch.nn.RNNBase subclasses into an 1d state vector.
    This is the reverse operation of default_unpack_hidden.

    :param hidden: unpacked hidden state, a tensor of num_recurrent_layers x batch_size x hidden_size
    :param num_recurrent_layers: number of recurrent layers
    :param hidden_size: size of the hidden layers (all equal)
    :param batch_size: if not none, the result should be 2d, and the first dimension represents parts of a data batch
    :return: packed hidden state.
    """
    if batch_size is None:
        # Simply flatten the hidden state
        return hidden.view(num_recurrent_layers * hidden_size)
    else:
        # Need to make sure that the batch dimension is the first element
        return hidden.permute(1, 0, 2).reshape(batch_size, num_recurrent_layers * hidden_size)
