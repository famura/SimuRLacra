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
from torch.nn.utils import convert_parameters as cp
from typing import Sequence, Callable, Iterable

import pyrado
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.utils.data_types import EnvSpec
from pyrado.policies.base import Policy
from pyrado.policies.initialization import init_param
from pyrado.utils.tensor import atleast_2D


class FNN(nn.Module):
    """ Feed-forward neural network """

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_sizes: Sequence[int],
                 hidden_nonlin: [Callable, Sequence[Callable]],
                 dropout: float = 0.,
                 output_nonlin: Callable = None,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param input_size: number of inputs
        :param output_size: number of outputs
        :param hidden_sizes: sizes of hidden layers (every entry creates one hidden layer)
        :param hidden_nonlin: nonlinearity for hidden layers
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param output_nonlin: nonlinearity for output layer
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        self._device = 'cuda' if use_cuda and to.cuda.is_available() else 'cpu'

        super().__init__()  # init nn.Module

        # Store settings
        # TODO One day replace legacy above with below
        self.hidden_nonlin = hidden_nonlin if isinstance(hidden_nonlin, Iterable) else len(hidden_sizes)*[hidden_nonlin]
        self.dropout = dropout
        self.output_nonlin = output_nonlin

        # Create hidden layers (stored in ModuleList so their parameters are tracked)
        self.hidden_layers = nn.ModuleList()
        last_size = input_size
        for hs in hidden_sizes:
            self.hidden_layers.append(nn.Linear(last_size, hs))
            # Current output size is next layer input size
            last_size = hs
            # Add a dropout layer after every hidden layer
            if self.dropout > 0:
                self.hidden_layers.append(nn.Dropout(p=self.dropout))

        # Create output layer
        self.output_layer = nn.Linear(last_size, output_size)

        # Initialize parameter values
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    @property
    def device(self) -> str:
        """ Get the device (CPU or GPU) on which the FNN is stored. """
        return self._device

    @property
    def param_values(self) -> to.Tensor:
        """
        Get the parameters of the policy as 1d array.
        The values are copied, modifying the return value does not propagate to the actual policy parameters.
        """
        return cp.parameters_to_vector(self.parameters())

    @param_values.setter
    def param_values(self, param):
        """ Set the policy parameters from an 1d array. """
        cp.vector_to_parameters(param, self.parameters())

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        """
        Initialize the network's parameters. By default the parameters are initialized randomly.

        :param init_values: Tensor of fixed initial network parameter values
        """
        if init_values is None:
            # Initialize hidden layers
            for i, layer in enumerate(self.hidden_layers):
                if self.dropout == 0:
                    # If there is no dropout, initialize weights and biases for every layer
                    init_param(layer, **kwargs)
                elif self.dropout > 0 and i%2 == 0:
                    # If there is dropout, omit the initialization for the dropout layers
                    init_param(layer, **kwargs)

            # Initialize output layer
            init_param(self.output_layer, **kwargs)

        else:
            self.param_values = init_values

    def forward(self, input: to.Tensor) -> to.Tensor:
        input = input.to(self.device)
        # Pass input through hidden layers
        next_input = input

        for i, layer in enumerate(self.hidden_layers):
            next_input = layer(next_input)
            # Apply non-linearity if any
            if self.dropout == 0:
                # If there is no dropout, apply the nonlinearity to every layer
                if self.hidden_nonlin[i] is not None:
                    next_input = self.hidden_nonlin[i](next_input)
            elif self.dropout > 0 and i%2 == 0:
                # If there is dropout, only apply the nonlinearity to every second layer
                if self.hidden_nonlin[i//2] is not None:
                    next_input = self.hidden_nonlin[i//2](next_input)

        # And through the output layer
        output = self.output_layer(next_input)
        if self.output_nonlin is not None:
            output = self.output_nonlin(output)

        return output


class FNNPolicy(Policy):
    """ Feed-forward neural network policy """

    name: str = 'fnn'

    def __init__(self,
                 spec: EnvSpec,
                 hidden_sizes: Sequence[int],
                 hidden_nonlin: [Callable, Sequence[Callable]],
                 dropout: float = 0.,
                 output_nonlin: Callable = None,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: environment specification
        :param hidden_sizes: sizes of hidden layer outputs. Every entry creates one hidden layer.
        :param hidden_nonlin: nonlinearity for hidden layers
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param output_nonlin: nonlinearity for output layer
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, use_cuda)

        # Create the feed-forward neural network
        self.net = FNN(
            input_size=spec.obs_space.flat_dim,
            output_size=spec.act_space.flat_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlin=hidden_nonlin,
            dropout=dropout,
            output_nonlin=output_nonlin,
            use_cuda=use_cuda)

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.net.init_param(None, **init_param_kwargs)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Forward to FNN's initialization function
            self.net.init_param(init_values, **kwargs)
        else:
            self.param_values = init_values

    def forward(self, obs: to.Tensor) -> to.Tensor:
        # Get the action from the owned FNN
        return self.net(obs)


class DiscrActQValFNNPolicy(Policy):
    """ State-action value (Q-value) feed-forward neural network policy for discrete actions """

    name: str = 'd_fnn'

    def __init__(self,
                 spec: EnvSpec,
                 hidden_sizes: Sequence[int],
                 hidden_nonlin: [Callable, Sequence[Callable]],
                 dropout: float = 0.,
                 output_nonlin: Callable = None,
                 init_param_kwargs: dict = None):
        """
        Constructor

        :param spec: environment specification
        :param hidden_sizes: sizes of hidden layer outputs. Every entry creates one hidden layer.
        :param hidden_nonlin: nonlinearity for hidden layers
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param output_nonlin: nonlinearity for output layer
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        """
        if not isinstance(spec.act_space, DiscreteSpace):
            raise pyrado.TypeErr(given=spec.act_space, expected_type=DiscreteSpace)
        super().__init__(spec, use_cuda=to.cuda.is_available())

        # Create the feed-forward neural network
        self.net = FNN(
            input_size=spec.obs_space.flat_dim + spec.act_space.ele_dim,
            output_size=1,
            hidden_sizes=hidden_sizes,
            hidden_nonlin=hidden_nonlin,
            dropout=dropout,
            output_nonlin=output_nonlin)

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.net.init_param(None, **init_param_kwargs)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is None:
            # Forward to FNN's initialization function
            self.net.init_param(init_values, **kwargs)
        else:
            self.param_values = init_values

    def q_values(self, obs: to.Tensor) -> to.Tensor:
        """
        Compute the state-action values for the given observations and all possible actions.
        Since we operate on a discrete ation space, we can construct a table
        o1 a1
        o1 a2
         ...
        oN a1
        oN a2

        :param obs: obersvations
        :return: Q-values for all state-action combinations, dim = batch_size x act_space_flat_sim
        """
        # Create batched state-action table
        obs = atleast_2D(obs)  # btach dim is along first axis
        columns_obs = obs.repeat_interleave(repeats=self.env_spec.act_space.flat_dim, dim=0)
        columns_act = to.tensor(self.env_spec.act_space.eles).repeat(obs.shape[0], 1)

        # Batch process via Pytorch Module class
        q_vals = self.net(to.cat([columns_obs, columns_act], dim=1))

        # Return reshaped tensor (different actions are over columns)
        return q_vals.view(-1, self.env_spec.act_space.flat_dim)

    def q_values_chosen(self, obs: to.Tensor) -> to.Tensor:
        """
        Compute the state-action values for the given observations and all possible actions.
        Since we operate on a discrete ation space, we can construct a table.

        :param obs: obersvations
        :return: Q-values for all state-action combinations, dimension equals flat action space dimension
        """

        # Get the Q-values from the owned FNN
        obs = atleast_2D(obs)
        q_vals = self.q_values(obs)

        # Select the Q-values from the that the policy would have selected
        act_idcs = to.argmax(q_vals, dim=1)
        return q_vals.gather(dim=1, index=act_idcs.view(-1, 1)).squeeze(1)  # select columns

    def forward(self, obs: to.Tensor) -> to.Tensor:
        # Get the Q-values from the owned FNN
        obs = atleast_2D(obs)
        batch_dim = obs.shape[0]
        q_vals = self.q_values(obs)

        # Select the actions with the highest Q-value
        act_idcs = to.argmax(q_vals, dim=1)
        all_acts = to.tensor(self.env_spec.act_space.eles).view(1, -1)  # get all possible (discrete) actions
        acts = all_acts.repeat(batch_dim, 1)
        if batch_dim == 1:
            return acts.gather(dim=1, index=act_idcs.view(-1, 1)).squeeze(0)  # select column
        if batch_dim > 1:
            return acts.gather(dim=1, index=act_idcs.view(-1, 1))  # select columns
