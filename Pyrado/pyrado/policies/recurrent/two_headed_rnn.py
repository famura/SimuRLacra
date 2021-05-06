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

from typing import Callable, Tuple

import torch as to
from torch import nn as nn
from torch.nn import RNNBase

import pyrado
from pyrado.policies.base import TwoHeadedPolicy
from pyrado.policies.initialization import init_param
from pyrado.policies.recurrent.base import RecurrentPolicy, default_pack_hidden, default_unpack_hidden
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


class TwoHeadedRNNPolicyBase(TwoHeadedPolicy, RecurrentPolicy):
    """
    Base class for recurrent policies, which are wrapping torch.nn.RNNBase, and have a common body and two heads that
    have a separate last layer
    """

    # Type of recurrent network. Is None in base class to force inheritance.
    recurrent_network_type: RNNBase = None

    def __init__(
        self,
        spec: EnvSpec,
        shared_hidden_size: int,
        shared_num_recurrent_layers: int,
        head_1_size: int = None,
        head_2_size: int = None,
        head_1_output_nonlin: Callable = None,
        head_2_output_nonlin: Callable = None,
        shared_dropout: float = 0.0,
        init_param_kwargs: dict = None,
        use_cuda: bool = False,
        **recurrent_net_kwargs,
    ):
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
        self.num_recurrent_layers = shared_num_recurrent_layers

        # Create RNN layers
        assert self.recurrent_network_type is not None, "Can not instantiate RNNPolicyBase!"
        self.shared = self.recurrent_network_type(  # pylint: disable=not-callable
            input_size=spec.obs_space.flat_dim,
            hidden_size=shared_hidden_size,
            num_layers=shared_num_recurrent_layers,
            bias=True,
            batch_first=False,
            dropout=shared_dropout,
            bidirectional=False,
            **recurrent_net_kwargs,
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
            # Initialize the layers using default initialization
            init_param(self.shared, **kwargs)
            init_param(self.head_1, **kwargs)
            init_param(self.head_2, **kwargs)
        else:
            self.param_values = init_values

    @property
    def hidden_size(self) -> int:
        # The total number of hidden parameters is the hidden layer size times the hidden layer count
        return self._hidden_size * self.num_recurrent_layers

    def forward(
        self, obs: to.Tensor, hidden: to.Tensor = None
    ) -> Tuple[to.Tensor, to.Tensor, to.Tensor]:  # pylint: disable=arguments-differ
        obs = obs.to(device=self.device, dtype=to.get_default_dtype())

        # Adjust the input's shape to be compatible with PyTorch's RNN implementation
        batch_size = None
        # We assume flattened observations, if they are 2d, they're batched.
        if len(obs.shape) == 1:
            obs = obs.view(1, 1, -1)
        elif len(obs.shape) == 2:
            batch_size = obs.shape[0]
            obs = obs.view(1, obs.shape[0], obs.shape[1])
        else:
            raise pyrado.ShapeErr(
                msg=f"Improper shape of 'obs'. Policy received {obs.shape}," f"but shape should be 1-dim or 2-dim"
            )

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

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = "hidden_states") -> Tuple[to.Tensor, to.Tensor]:
        if not rollout.data_format == "torch":
            raise pyrado.TypeErr(msg="The rollout rollout passed to evaluate() must be of type torch.Tensor!")
        if not rollout.continuous:
            raise pyrado.ValueErr(msg="The rollout rollout passed to evaluate() from a continuous rollout!")

        # Set policy, i.e. PyTorch nn.Module, to evaluation mode
        self.eval()

        act_list = []
        head2_list = []
        for ro in rollout.iterate_rollouts():
            if hidden_states_name in rollout.data_names:
                # Get initial hidden state from first step
                hidden = ro[0][hidden_states_name]
            else:
                # Let the network pick the default hidden state
                hidden = None
            # Run steps consecutively reusing the hidden state
            for step in ro:
                act, head2, hidden = self(step.observation, hidden)
                act_list.append(act)
                head2_list.append(head2)

        # Set policy, i.e. PyTorch nn.Module, back to training mode
        self.train()

        return to.stack(act_list), to.stack(head2_list)

    def _unpack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Unpack the flat hidden state vector into a form the actual network module can use.
        Since hidden usually comes from some outer source, this method should validate it's shape.
        The default implementation is defined by default_unpack_hidden.

        :param hidden: flat hidden state
        :param batch_size: if not None, hidden is 2-dim and the first dimension represents parts of a rollout batch
        :return: unpacked hidden state, ready for the network
        """
        return default_unpack_hidden(hidden, self.num_recurrent_layers, self._hidden_size, batch_size)

    def _pack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Pack the hidden state returned by the network into an 1-dim state vector.
        This should be the reverse operation of _unpack_hidden.
        The default implementation is defined by default_pack_hidden.

        :param hidden: hidden state as returned by the network
        :param batch_size: if not None, the result should be 2-dim and the first dimension represents parts of a rollout batch
        :return: packed hidden state
        """
        return default_pack_hidden(hidden, self.num_recurrent_layers, self._hidden_size, batch_size)


class TwoHeadedRNNPolicy(TwoHeadedRNNPolicyBase):
    """Two-headed policy backed by a multi-layer RNN"""

    name: str = "thrnn"

    recurrent_network_type = nn.RNN

    def __init__(
        self,
        spec: EnvSpec,
        shared_hidden_size: int,
        shared_num_recurrent_layers: int,
        shared_hidden_nonlin: str = "tanh",
        head_1_size: int = None,
        head_2_size: int = None,
        head_1_output_nonlin: Callable = None,
        head_2_output_nonlin: Callable = None,
        shared_dropout: float = 0.0,
        init_param_kwargs: dict = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param shared_hidden_size: size of the hidden layers (all equal)
        :param shared_num_recurrent_layers: number of recurrent layers
        :param shared_hidden_nonlin: nonlinearity for the shared hidden rnn layers, either 'tanh' or 'relu'
        :param head_1_size: size of the fully connected layer for head 1, if `None` this is set to the action space dim
        :param head_2_size: size of the fully connected layer for head 2, if `None` this is set to the action space dim
        :param head_1_output_nonlin: nonlinearity for output layer of the first head
        :param head_2_output_nonlin: nonlinearity for output layer of the second head
        :param shared_dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(
            spec,
            shared_hidden_size,
            shared_num_recurrent_layers,
            head_1_size,
            head_2_size,
            head_1_output_nonlin,
            head_2_output_nonlin,
            shared_dropout,
            init_param_kwargs,
            use_cuda,
            nonlinearity=shared_hidden_nonlin,  # pass as extra arg to RNN constructor, must be kwarg
        )


class TwoHeadedGRUPolicy(TwoHeadedRNNPolicyBase):
    """Two-headed policy backed by a multi-layer GRU"""

    name: str = "thgru"

    recurrent_network_type = nn.GRU


class TwoHeadedLSTMPolicy(TwoHeadedRNNPolicyBase):
    """Two-headed policy backed by a multi-layer LSTM"""

    name: str = "thlstm"

    recurrent_network_type = nn.LSTM

    @property
    def hidden_size(self) -> int:
        # LSTM has two hidden variables per layer
        return self.num_recurrent_layers * self._hidden_size * 2

    def _unpack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        # Special case - need to split into hidden and cell term memory
        # Assume it's a flattened view of hid/cell x nrl x batch x hs
        if len(hidden.shape) == 1:
            assert (
                hidden.shape[0] == self.hidden_size
            ), "Passed hidden variable's size doesn't match the one required by the network."
            # We could handle this case, but for now it's not necessary
            assert batch_size is None, "Cannot use batched observations with unbatched hidden state"

            # Reshape to hid/cell x nrl x batch x hs
            hd = hidden.view(2, self.num_recurrent_layers, 1, self._hidden_size)
            # Split hidden and cell state
            return hd[0, ...], hd[1, ...]

        elif len(hidden.shape) == 2:
            assert (
                hidden.shape[1] == self.hidden_size
            ), "Passed hidden variable's size doesn't match the one required by the network."
            assert (
                hidden.shape[0] == batch_size
            ), f"Batch size of hidden state ({hidden.shape[0]}) must match batch size of observations ({batch_size})"

            # Reshape to hid/cell x nrl x batch x hs
            hd = hidden.view(batch_size, 2, self.num_recurrent_layers, self._hidden_size).permute(1, 2, 0, 3)
            # Split hidden and cell state
            return hd[0, ...], hd[1, ...]

        else:
            raise pyrado.ShapeErr(
                msg=f"Improper shape of 'hidden'. Policy received {hidden.shape}," f"but shape should be 1- or 2-dim"
            )

    def _pack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        # Hidden is a tuple, need to turn it to stacked state
        stacked = to.stack(hidden)

        if batch_size is None:
            # Simply flatten
            return stacked.view(self.hidden_size)
        else:
            # We bring batch dimension to front
            return stacked.permute(2, 0, 1, 3).reshape(batch_size, self.hidden_size)
