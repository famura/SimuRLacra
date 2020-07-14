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
from typing import Sequence, Callable, Tuple

import pyrado
from pyrado.policies.base_recurrent import RecurrentPolicy
from pyrado.policies.fnn import FNN
from pyrado.policies.initialization import init_param
from pyrado.policies.rnn import default_unpack_hidden, default_pack_hidden
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy


class TwoHeadedPolicy(Policy, ABC):
    """ Base class for policies with a shared body and two separate heads. """

    @abstractmethod
    def init_param(self, init_values: to.Tensor = None, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, obs: to.Tensor) -> [to.Tensor, (to.Tensor, to.Tensor)]:
        raise NotImplementedError


class TwoHeadedFNNPolicy(TwoHeadedPolicy):
    """ Policy architecture which has a common body and two heads that have a separate last layer """

    name: str = '2h_fnn'

    def __init__(self,
                 spec: EnvSpec,
                 shared_hidden_sizes: Sequence[int],
                 shared_hidden_nonlin: [Callable, Sequence[Callable]],
                 head_1_size: int = None,
                 head_2_size: int = None,
                 head_1_output_nonlin: Callable = None,
                 head_2_output_nonlin: Callable = None,
                 shared_dropout: float = 0.,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param shared_hidden_sizes: sizes of shared hidden layer outputs. Every entry creates one shared hidden layer.
        :param shared_hidden_nonlin: nonlinearity for the shared hidden layers
        :param head_1_size: size of the fully connected layer for head 1, if `None` this is set to the action space dim
        :param head_2_size: size of the fully connected layer for head 2, if `None` this is set to the action space dim
        :param head_1_output_nonlin: nonlinearity for output layer of the first head
        :param head_2_output_nonlin: nonlinearity for output layer of the second head
        :param shared_dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        # Create the feed-forward neural network
        self.shared = FNN(
            input_size=spec.obs_space.flat_dim,
            output_size=shared_hidden_sizes[-1],
            hidden_sizes=shared_hidden_sizes,
            hidden_nonlin=shared_hidden_nonlin,
            dropout=shared_dropout,
            output_nonlin=None
        )

        # Create output layer
        head_1_size = spec.act_space.flat_dim if head_1_size is None else head_1_size
        head_2_size = spec.act_space.flat_dim if head_2_size is None else head_2_size
        self.head_1 = nn.Linear(shared_hidden_sizes[-1], head_1_size)
        self.head_2 = nn.Linear(shared_hidden_sizes[-1], head_2_size)
        self.head_1_output_nonlin = head_1_output_nonlin
        self.head_2_output_nonlin = head_2_output_nonlin

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            self.shared.init_param(None, **kwargs)
            init_param(self.head_1, **kwargs)
            init_param(self.head_2, **kwargs)
        else:
            self.param_values = init_values

    def forward(self, obs: to.Tensor) -> Tuple[to.Tensor, to.Tensor]:
        obs = obs.to(self.device)

        # Get the output of the last shared layer and pass this to the two headers separately
        x = self.shared(obs)
        output_1 = self.head_1(x)
        output_2 = self.head_2(x)
        if self.head_1_output_nonlin is not None:
            output_1 = self.head_1_output_nonlin(output_1)
        if self.head_2_output_nonlin is not None:
            output_2 = self.head_2_output_nonlin(output_2)
        return output_1, output_2


class TwoHeadedGRUPolicy(TwoHeadedPolicy, RecurrentPolicy):
    """ Policy architecture which has a common body and two heads that have a separate last layer """

    name: str = '2h_gru'

    def __init__(self,
                 spec: EnvSpec,
                 shared_hidden_size: int,
                 shared_num_recurrent_layers: int,
                 head_1_size: int = None,
                 head_2_size: int = None,
                 head_1_output_nonlin: Callable = None,
                 head_2_output_nonlin: Callable = None,
                 shared_dropout: float = 0.,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param shared_hidden_size: size of the hidden layers (all equal)
        :param shared_num_recurrent_layers: number of recurrent layers
        :param head_1_size: size of the fully connected layer for head 1, if `None` this is set to the action space dim
        :param head_2_size: size of the fully connected layer for head 2, if `None` this is set to the action space dim
        :param head_1_output_nonlin: nonlinearity for output layer of the first head
        :param head_2_output_nonlin: nonlinearity for output layer of the second head
        :param shared_dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        self._hidden_size = shared_hidden_size
        self._num_recurrent_layers = shared_num_recurrent_layers

        # Create the feed-forward neural network
        self.shared = nn.GRU(
            input_size=spec.obs_space.flat_dim,
            hidden_size=shared_hidden_size,
            num_layers=shared_num_recurrent_layers,
            bias=True,
            batch_first=False,
            dropout=shared_dropout,
            bidirectional=False,
        )

        # Create output layer
        self.head_1_size = spec.act_space.flat_dim if head_1_size is None else head_1_size
        self.head_2_size = spec.act_space.flat_dim if head_2_size is None else head_2_size
        self.head_1 = nn.Linear(shared_hidden_size, self.head_1_size)
        self.head_2 = nn.Linear(shared_hidden_size, self.head_2_size)
        self.head_1_output_nonlin = head_1_output_nonlin
        self.head_2_output_nonlin = head_2_output_nonlin

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            init_param(self.shared, **kwargs)
            init_param(self.head_1, **kwargs)
            init_param(self.head_2, **kwargs)
        else:
            self.param_values = init_values

    @property
    def hidden_size(self) -> int:
        # The total number of hidden parameters is the hidden layer size times the hidden layer count
        return self._hidden_size*self._num_recurrent_layers

    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> Tuple[to.Tensor, to.Tensor, to.Tensor]:
        obs = obs.to(self.device)

        # Adjust the input's shape to be compatible with PyTorch's RNN implementation
        batch_size = None
        # We assume flattened observations, if they are 2d, they're batched.
        if len(obs.shape) == 1:
            obs = obs.view(1, 1, -1)
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
            obs = obs.view(1, obs.shape[0], obs.shape[1])
        else:
            raise pyrado.ShapeErr(msg=f"Improper shape of 'obs'. Policy received {obs.shape},"
                                      f"but shape should be 1-dim or 2-dim")

        # Unpack hidden tensor if specified
        # The network can handle getting None by using default values
        if hidden is not None:
            hidden = self._unpack_hidden(hidden, batch_size)

        # Get the output of the last shared layer and pass this to the two headers separately
        x, new_hidden = self.shared(obs, hidden)

        # And through the two output layers
        output_1 = self.head_1(x)
        output_2 = self.head_2(x)
        if self.head_1_output_nonlin is not None:
            output_1 = self.head_1_output_nonlin(output_1)
        if self.head_2_output_nonlin is not None:
            output_2 = self.head_2_output_nonlin(output_2)

        # Adjust the outputs' shape to be compatible with the space of the environment
        if batch_size is None:
            # One sample in, one sample out
            output_1 = output_1.view(self.head_1_size)
            output_2 = output_2.view(self.head_2_size)
        else:
            # Return a batch if a batch was passed
            output_1 = output_1.view(batch_size, self.head_1_size)
            output_2 = output_2.view(batch_size, self.head_2_size)

        new_hidden = self._pack_hidden(new_hidden, batch_size)

        return output_1, output_2, new_hidden

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = 'hidden_states') -> Tuple[to.Tensor, to.Tensor]:
        act_list = []
        head2_list = []
        for ro in rollout.iterate_rollouts():
            if hidden_states_name in rollout.data_names:
                # Get initial hidden state from first step
                hs = ro[0][hidden_states_name]
            else:
                # Let the network pick the default hidden state.
                hs = None

            # Run each step separately
            for step in ro:
                act, head2, hs = self(step.observation, hs)
                act_list.append(act)
                head2_list.append(head2)

        return to.stack(act_list), to.stack(head2_list)

    def _unpack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Unpack the flat hidden state vector into a form the actual network module can use.
        Since hidden usually comes from some outer source, this method should validate it's shape.
        The default implementation is defined by default_unpack_hidden.

        :param hidden: flat hidden state
        :param batch_size: if not None, hidden is 2-dim and the first dimension represents parts of a data batch
        :return: unpacked hidden state, ready for the network
        """
        return default_unpack_hidden(hidden, self._num_recurrent_layers, self._hidden_size, batch_size)

    def _pack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Pack the hidden state returned by the network into an 1-dim state vector.
        This should be the reverse operation of _unpack_hidden.
        The default implementation is defined by default_pack_hidden.

        :param hidden: hidden state as returned by the network
        :param batch_size: if not None, the result should be 2-dim and the first dimension represents parts of a data batch
        :return: packed hidden state
        """
        return default_pack_hidden(hidden, self._num_recurrent_layers, self._hidden_size, batch_size)
