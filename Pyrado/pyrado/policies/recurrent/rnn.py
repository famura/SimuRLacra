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

from typing import Callable

import torch as to
import torch.nn as nn

import pyrado
from pyrado.policies.initialization import init_param
from pyrado.policies.recurrent.base import RecurrentPolicy, default_pack_hidden, default_unpack_hidden
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import EnvSpec


class RNNPolicyBase(RecurrentPolicy):
    """ Base class for recurrent policies wrapping torch.nn.RNNBase subclasses """

    # Type of recurrent network. Is None in base class to force inheritance.
    recurrent_network_type = None

    def __init__(
        self,
        spec: EnvSpec,
        hidden_size: int,
        num_recurrent_layers: int,
        output_nonlin: Callable = None,
        dropout: float = 0.0,
        init_param_kwargs: dict = None,
        use_cuda: bool = False,
        **recurrent_net_kwargs,
    ):
        """
        Constructor

        :param spec: environment specification
        :param hidden_size: size of the hidden layers (all equal)
        :param num_recurrent_layers: number of equally sized hidden layers
        :param output_nonlin: nonlinearity for output layer
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param recurrent_net_kwargs: any extra kwargs are passed to the recurrent net's constructor
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        self._hidden_size = hidden_size
        self.num_recurrent_layers = num_recurrent_layers

        # Create RNN layers
        assert self.recurrent_network_type is not None, "Can not instantiate RNNPolicyBase!"
        self.rnn_layers = self.recurrent_network_type(
            input_size=spec.obs_space.flat_dim,
            hidden_size=hidden_size,
            num_layers=num_recurrent_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=False,
            **recurrent_net_kwargs,
        )

        # Create output layer
        self.output_layer = nn.Linear(hidden_size, spec.act_space.flat_dim)
        self.output_nonlin = output_nonlin

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Initialize the layers using default initialization
            init_param(self.rnn_layers, **kwargs)
            init_param(self.output_layer, **kwargs)
        else:
            self.param_values = init_values

    @property
    def hidden_size(self) -> int:
        # The total number of hidden parameters is the hidden layer size times the hidden layer count
        return self.num_recurrent_layers * self._hidden_size

    def forward(self, obs: to.Tensor, hidden: to.Tensor = None) -> (to.Tensor, to.Tensor):
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

        # Unpack hidden tensor if specified. The network can handle getting None by using default values.
        if hidden is not None:
            hidden = hidden.to(device=self.device, dtype=to.get_default_dtype())
            hidden = self._unpack_hidden(hidden, batch_size)

        # Pass the input through hidden RNN layers
        out, new_hidden = self.rnn_layers(obs, hidden)

        # And through the output layer
        act = self.output_layer(out)
        if self.output_nonlin is not None:
            act = self.output_nonlin(act)

        # Adjust the outputs' shape to be compatible with the space of the environment
        if batch_size is None:
            # One sample in, one sample out
            act = act.view(self.env_spec.act_space.flat_dim)
        else:
            # Return a batch if a batch was passed
            act = act.view(batch_size, self.env_spec.act_space.flat_dim)
        new_hidden = self._pack_hidden(new_hidden, batch_size)

        return act, new_hidden

    def evaluate(self, rollout: StepSequence, hidden_states_name: str = "hidden_states") -> to.Tensor:
        if not rollout.data_format == "torch":
            raise pyrado.TypeErr(msg="The rollout data passed to evaluate() must be of type torch.Tensor!")
        if not rollout.continuous:
            raise pyrado.ValueErr(msg="The rollout data passed to evaluate() from a continuous rollout!")

        # Set policy, i.e. PyTorch nn.Module, to evaluation mode
        self.eval()

        # The passed sample collection might contain multiple rollouts.
        # Note:
        # While we *could* try to convert this to a PackedSequence, allowing us to only call the network once, that
        # would require a lot of reshaping on the result. So let's not. If performance becomes an issue, revisit here.
        act_list = []
        for ro in rollout.iterate_rollouts():
            if hidden_states_name in rollout.data_names:
                # Get initial hidden state from first step
                hidden = self._unpack_hidden(ro[0][hidden_states_name])
            else:
                # Let the network pick the default hidden state
                hidden = None

            # Reshape observations to match PyTorch's RNN sequence protocol
            obs = ro.get_data_values("observations", True).unsqueeze(1)
            obs = obs.to(device=self.device, dtype=to.get_default_dtype())

            # Pass the input through hidden RNN layers
            out, _ = self.rnn_layers(obs, hidden)

            # And through the output layer
            act = self.output_layer(out.squeeze(1))
            if self.output_nonlin is not None:
                act = self.output_nonlin(act)

            # Collect the actions
            act_list.append(act)

        # Set policy, i.e. PyTorch nn.Module, back to training mode
        self.train()

        return to.cat(act_list)

    def _unpack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Unpack the flat hidden state vector into a form the actual network module can use.
        Since hidden usually comes from some outer source, this method should validate it's shape.
        The default implementation is defined by `default_unpack_hidden`.

        :param hidden: flat hidden state
        :param batch_size: if not `None`, hidden is 2-dim and the first dim represents parts of a data batch
        :return: unpacked hidden state, ready for the network
        """
        return default_unpack_hidden(hidden, self.num_recurrent_layers, self._hidden_size, batch_size)

    def _pack_hidden(self, hidden: to.Tensor, batch_size: int = None):
        """
        Pack the hidden state returned by the network into an 1-dim state vector.
        This should be the reverse operation of `_unpack_hidden`.
        The default implementation is defined by `default_pack_hidden`.

        :param hidden: hidden state as returned by the network
        :param batch_size: if not `None`, the result should be 2-dim and the first dim represents parts of a data batch
        :return: packed hidden state
        """
        return default_pack_hidden(hidden, self.num_recurrent_layers, self._hidden_size, batch_size)


class RNNPolicy(RNNPolicyBase):
    """ Policy backed by a multi-layer RNN """

    name: str = "rnn"

    recurrent_network_type = nn.RNN

    def __init__(
        self,
        spec: EnvSpec,
        hidden_size: int,
        num_recurrent_layers: int,
        hidden_nonlin: str = "tanh",
        output_nonlin: Callable = None,
        dropout: float = 0.0,
        init_param_kwargs: dict = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param hidden_size: size of the hidden layers (all equal)
        :param num_recurrent_layers: number of equally sized hidden layers
        :param hidden_nonlin: nonlinearity for the hidden rnn layers, either 'tanh' or 'relu'
        :param output_nonlin: nonlinearity for output layer
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(
            spec,
            hidden_size,
            num_recurrent_layers,
            output_nonlin,
            dropout,
            init_param_kwargs,
            use_cuda,
            nonlinearity=hidden_nonlin,  # pass as extra arg to RNN constructor, must be kwarg
        )


class GRUPolicy(RNNPolicyBase):
    """ Policy backed by a multi-layer GRU """

    name: str = "gru"

    recurrent_network_type = nn.GRU


class LSTMPolicy(RNNPolicyBase):
    """ Policy backed by a multi-layer LSTM """

    name: str = "lstm"

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
