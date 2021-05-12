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
from typing import List, Optional, Tuple, Union
from warnings import warn

import torch as to
import torch.nn as nn
from dtw import dtw
from dtw.stepPattern import rabinerJuangStepPattern
from torch.nn.utils import convert_parameters as cp
from torch.utils.data import DataLoader, Dataset

import pyrado
from pyrado.policies.initialization import init_param
from pyrado.utils.data_types import EnvSpec, merge_dicts


# class SingleTensorDataset(Dataset[to.Tensor]):
#     """
#     Custom subclass of the`Dataset` class wrapping a single PyTorch tensor
#
#     .. note::
#         Use in combination with `collate_fn=lambda x: x` in the `DataLoader` constructor.
#     """
#
#     tensor: to.Tensor
#
#     def __init__(self, tensor: to.Tensor) -> None:
#         """
#         Constructor
#
#         :param tensor: input tensor of at least 2 dimensions
#         """
#         if not isinstance(tensor, to.Tensor):
#             raise pyrado.TypeErr(given=tensor, expected_type=to.Tensor)
#         if not tensor.ndim == 4:
#             raise pyrado.ShapeErr(
#                 msg=f"The input tensor must at have exactly 4 dimensions, but its shape is {tensor.shape}!"
#             )
#         self.tensor = tensor.reshape(-1, tensor.shape[2], tensor.shape[3])
#
#     def __getitem__(self, idx: int):
#         """ Get a slice though the 1-st dimension, i.e. the combined batch and rollout dimension. """
#         return self.tensor[idx]
#
#     def __len__(self):
#         """ Get the length of the data set, i.e. batch_size * num_rollouts. """
#         return self.tensor.shape[0]


class Embedding(ABC, nn.Module):
    """
    Base class for all embeddings used for simulation-based inference with time series data

    .. note::
        The features of each rollout are concatenated, and since the inference procedure requires a consistent size of
        the inputs, it is necessary that all rollouts yield the same number of features, i.e. have equal length!
    """

    name: str
    requires_target_domain_data: bool

    def __init__(
        self,
        spec: EnvSpec,
        dim_data: int,
        downsampling_factor: int = 1,
        idcs_data: Optional[Union[Tuple[int], List[int]]] = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param dim_data: number of dimensions of one data sample, i.e. one time step. By default, this is the sum of the
                         states and action spaces' flat dimensions. This number is doubled if the embedding
                         target domain data.
        :param downsampling_factor: skip evey `downsampling_factor` time series sample, no downsampling by default
        :param idcs_data: list or tuple of integers to select specific states from the data (always using all actions),
                          by default `None` to select all states
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(spec, EnvSpec):
            raise pyrado.TypeErr(given=spec, expected_type=EnvSpec)
        if not isinstance(dim_data, int) or dim_data < 1:
            raise pyrado.ValueErr(given=dim_data, ge_constraint="1 (int)")
        if not isinstance(downsampling_factor, int) or downsampling_factor < 1:
            raise pyrado.ValueErr(given=downsampling_factor, ge_constraint="1 (int)")

        super().__init__()

        self._env_spec = spec
        self._dim_data_orig = 2 * dim_data if self.requires_target_domain_data else dim_data
        self._idcs_data = tuple(idcs_data) if idcs_data is not None else None
        self.downsampling_factor = downsampling_factor

        # Manage device
        if not use_cuda:
            self._device = "cpu"
        elif use_cuda and to.cuda.is_available():
            self._device = "cuda"
        elif use_cuda and not to.cuda.is_available():
            warn("Tried to run on CUDA, but it is not available. Falling back to CPU.")
            self._device = "cpu"
        self.to(self.device)

    @property
    @abstractmethod
    def dim_output(self) -> int:
        """Get the dimension of the embeddings output, i.e. its feature dimension."""
        raise NotImplementedError

    @property
    def device(self) -> str:
        """Get the device (CPU or GPU) on which the embedding is stored."""
        return self._device

    @abstractmethod
    def summary_statistic(self, data: to.Tensor) -> to.Tensor:
        raise NotImplementedError

    @staticmethod
    def pack(data: to.Tensor) -> to.Tensor:
        """
        Reshape the data such that the shape is [batch_dim, num_rollouts, data_points_flattened].

        :param data: un-packed a.k.a. un-flattened data
        :return: packed a.k.a. flattened data
        """
        if data.ndim == 2:
            # The data is not batched, and we have one target domain rollouts which is un-flattened
            return data.view(1, 1, -1)

        elif data.ndim == 3:
            # The data is not batched, but we have multiple target domain rollouts which are un-flattened
            num_rollouts = data.shape[0]
            return data.view(1, num_rollouts, -1)

        elif data.ndim == 4:
            # The data is batched, and we have multiple target domain rollouts
            batch_size, num_rollouts = data.shape[:2]
            return data.view(batch_size, num_rollouts, -1)

        else:
            raise pyrado.ShapeErr(msg=f"The data must have either 2, 3, or 4 dimensions, not {data.ndim}!")

    @staticmethod
    def unpack(data: to.Tensor, dim_data_orig: int) -> to.Tensor:
        """
        Reshape the data such that the shape is [batch_dim, num_rollouts, len_time_series, dim_data].

        :param data: packed a.k.a. flattened data
        :param dim_data_orig: dimension of the original data
        :return: un-pack a.k.a. un-flattened data
        """
        if data.ndim != 3:
            raise pyrado.ShapeErr(
                msg=f"The data must have exactly 3 dimensions, but is of shape {data.shape}! Check if packed before "
                f"unpacking. This error can also occur if the simulator is not batched. Either enable it to process "
                f"batches of domain parameters or implement a 2-dim case of pack() and unpack()."
            )

        batch_size, num_rollouts = data.shape[:2]  # packing is designed to ensure this
        data = data.view(batch_size, num_rollouts, -1, dim_data_orig)

        if data.ndim != 4:
            raise pyrado.ShapeErr(msg="The data tensor must have exactly 4 dimensions after unpacking!")

        return data

    def forward_one_batch(self, data_batch: to.Tensor) -> to.Tensor:
        """
        Iterate over all rollouts and compute the features for each rollout separately, then average the features
        over the rollouts.

        :param data_batch: data batch of shape [num_rollouts, len_time_series, dim_data]
        :return: concatenation of the features for each rollout
        """
        return to.cat([self.summary_statistic(d) for d in data_batch], dim=0)

    def forward(self, data: to.Tensor) -> to.Tensor:
        """
        Transforms rollouts into the observations used for likelihood-free inference.
        Currently a state-representation as well as state-action summary-statistics are available.

        :param data: packed data of shape [batch_size, num_rollouts, len_time_series, dim_data]
        :return: features of the data extracted from the embedding of shape [[batch_size, num_rollouts * dim_feat]
        """
        data = data.to(device=self.device, dtype=to.get_default_dtype())

        # Bring the data back into the un-flattened form of shape [batch_size, num_rollouts, len_time_series, dim_data]
        data = Embedding.unpack(data, self._dim_data_orig)

        if self.downsampling_factor > 1:
            data = data[:, :, :: self.downsampling_factor, :]

        # Iterate over all data batches computing the features from the data
        x = to.stack([self.forward_one_batch(batch) for batch in data], dim=0)

        # Check the shape
        if x.shape != (data.shape[0], data.shape[1] * self.dim_output):
            raise pyrado.ShapeErr(given=x, expected_match=(data.shape[0], data.shape[1] * self.dim_output))

        return x


class LastStepEmbedding(Embedding):
    """
    Embedding for simulation-based inference with time series data which selects the last state of the rollouts
    """

    name: str = "lsemb"
    requires_target_domain_data: bool = False

    @property
    def dim_output(self) -> int:
        if self._idcs_data is not None:
            return len(self._idcs_data)
        else:
            return self._env_spec.state_space.flat_dim

    @to.no_grad()
    def summary_statistic(self, data: to.Tensor) -> to.Tensor:
        """
        Returns the last state of the rollout as a vector.

        :param data: states and actions of a rollout or segment to be transformed for inference
        :return: last states as a vector
        """
        if self._idcs_data is not None:
            last_state = data[-1, self._idcs_data]
        else:
            last_state = data[-1, : self._env_spec.state_space.flat_dim]
        return last_state.reshape(-1)


class DeltaStepsEmbedding(Embedding):
    """
    Embedding for simulation-based inference with time series data which returns the change in the states between
    consecutive time steps of the rollouts
    """

    name: str = "dsemb"
    requires_target_domain_data: bool = False

    def __init__(
        self,
        spec: EnvSpec,
        dim_data: int,
        len_rollouts: int,
        downsampling_factor: int = 1,
        idcs_data: Optional[Union[Tuple[int], List[int]]] = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param dim_data: number of dimensions of one data sample, i.e. one time step. By default, this is the sum of the
                         state and action spaces' flat dimensions. This number is doubled if the embedding
                         target domain data.
        :param len_rollouts: number of time steps per rollout without considering a potential downsampling later
                             (must be the same for all rollouts)
        :param downsampling_factor: skip evey `downsampling_factor` time series sample, the downsampling is done in the
                                    base class before calling `summary_statistic()`
        :param idcs_data: list or tuple of integers to select specific states from the data (always using all actions),
                          by default `None` to select all states
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(len_rollouts, int) or len_rollouts < 0:
            raise pyrado.ValueErr(given=len_rollouts, eq_constraint="1 (int)")

        super().__init__(spec, dim_data, downsampling_factor, idcs_data, use_cuda)

        self._len_rollouts = len_rollouts // downsampling_factor
        self.to(self.device)

    @property
    def dim_output(self) -> int:
        if self._idcs_data is not None:
            return (self._len_rollouts - 1) * len(self._idcs_data)
        else:
            return (self._len_rollouts - 1) * self._env_spec.state_space.flat_dim

    @to.no_grad()
    def summary_statistic(self, data: to.Tensor) -> to.Tensor:
        """
        Returns the last states of the rollout as a vector.

        :param data: states and actions of a rollout or segment to be transformed for inference
        :return: all states as a flattened vector
        """
        if data.shape[0] < 2:
            raise pyrado.ShapeErr(msg="The data tensor needs to contain at least two samples!")

        # Extract the states, and compute the deltas
        if self._idcs_data is not None:
            states = data[:, self._idcs_data]
        else:
            states = data[:, : self._env_spec.state_space.flat_dim]

        states_diff = states[1:] - states[:-1]
        return states_diff.reshape(-1)


class BayesSimEmbedding(Embedding):
    """
    Embedding for simulation-based inference with time series data which computes the same features of the rollouts
    states and actions as done in [1]

    [1] F. Ramos, R.C. Possas, D. Fox, "BayesSim: adaptive domain randomization via probabilistic inference for
        robotics simulators", RSS, 2019
    """

    name: str = "bsemb"
    requires_target_domain_data: bool = False

    @property
    def dim_output(self) -> int:
        if self._idcs_data is not None:
            state_dim = len(self._idcs_data)
        else:
            state_dim = self._env_spec.state_space.shape[0]
        act_dim = self._env_spec.act_space.shape[0]  # always use the full action
        return state_dim * act_dim + 2 * state_dim

    @to.no_grad()
    def summary_statistic(self, data: to.Tensor) -> to.Tensor:
        """
        Computing summary statistics based on approach in [1], see eq. (22).
        This method guarantees output which has the same size for every trajectory.

        [1] F. Ramos, R.C. Possas, D. Fox, "BayesSim: adaptive domain randomization via probabilistic inference for
            robotics simulators", RSS, 2019

        :param data: states and actions of a rollout or segment to be transformed for inference
        :return: summary statistics of the rollout
        """
        if data.shape[0] < 2:
            raise pyrado.ShapeErr(msg="The data tensor needs to contain at least two samples!")

        # Extract the states and actions from the data
        if self._idcs_data is not None:
            state = data[:, self._idcs_data]
        else:
            state = data[:, : self._env_spec.state_space.flat_dim]
        act = data[:-1, self._env_spec.state_space.flat_dim :]  # need to cut off one act due to using the delta state
        state_diff = state[1:] - state[:-1]

        # Compute the statistics
        act_state_dot_prod = to.einsum("ij,ik->jk", act, state_diff).view(-1)
        mean_state_diff = to.mean(state_diff, dim=0)
        var_state_diff = to.mean((mean_state_diff - state_diff) ** 2, dim=0)

        # Combine all the statistics
        return to.cat((act_state_dot_prod, mean_state_diff, var_state_diff), dim=0)


class RNNEmbedding(Embedding):
    """
    Embedding for simulation-based inference with time series data which uses an recurrent neural network, e.g. RNN,
    LSTM, or GRU, to compute features of the rollouts
    """

    name: str = "rnnemb"
    requires_target_domain_data: bool = False

    def __init__(
        self,
        spec: EnvSpec,
        dim_data: int,
        hidden_size: int,
        num_recurrent_layers: int,
        output_size: int,
        recurrent_network_type: type = nn.RNN,
        only_last_output: bool = False,
        len_rollouts: int = None,
        dropout: float = 0.0,
        init_param_kwargs: Optional[dict] = None,
        downsampling_factor: int = 1,
        idcs_data: Optional[Union[Tuple[int], List[int]]] = None,
        use_cuda: bool = False,
        **recurrent_net_kwargs,
    ):
        """
        Constructor

        :param spec: environment specification
        :param dim_data: number of dimensions of one data sample, i.e. one time step. By default, this is the sum of the
                         state and action spaces' flat dimensions. This number is doubled if the embedding
                         target domain data.
        :param hidden_size: size of the hidden layers (all equal)
        :param num_recurrent_layers: number of equally sized hidden layers
        :param recurrent_network_type: PyTorch recurrent network class, e.g. `nn.RNN`, `nn.LSTM`, or `nn.GRU`
        :param output_size: size of the features at every time step, which are eventually reshaped into a vector
        :param only_last_output: if `True`, only the last output of the network is used as a feature for sbi, else
                                 there will be an output every `downsampling_factor` time steps. Moreover, if `True` the
                                 constructor does not need to know how long the rollouts are.
        :param len_rollouts: number of time steps per rollout without considering a potential downsampling later
                             (must be the same for all rollouts)
        :param dropout: dropout probability, default = 0 deactivates dropout
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param recurrent_net_kwargs: any extra kwargs are passed to the recurrent net's constructor
        :param downsampling_factor: skip evey `downsampling_factor` time series sample, the downsampling is done in the
                                    base class before calling `summary_statistic()`
        :param idcs_data: list or tuple of integers to select specific states from the data (always using all actions),
                          by default `None` to select all states
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(spec, dim_data, downsampling_factor, idcs_data, use_cuda)

        # Check the time sequence length if necessary
        if not only_last_output:
            if not isinstance(len_rollouts, int) or len_rollouts < 0:
                raise pyrado.ValueErr(given=len_rollouts, eq_constraint="1 (int)")
            self._len_rollouts = len_rollouts // downsampling_factor
        else:
            self._len_rollouts = None  # use to signal only_last_output == True

        if recurrent_network_type == nn.RNN:
            recurrent_net_kwargs = merge_dicts([dict(nonlinearity="tanh"), recurrent_net_kwargs])

        # Create the RNN layers
        self.rnn_layers = recurrent_network_type(
            input_size=dim_data if idcs_data is None else len(idcs_data),
            hidden_size=hidden_size,
            num_layers=num_recurrent_layers,
            bias=True,
            batch_first=False,
            dropout=dropout,
            bidirectional=False,
            **recurrent_net_kwargs,
        )

        # Create the output layer
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize parameter values
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

        # Detach the complete network, i.e. use it with the random initialization
        for p in self.parameters():
            p.requires_grad = False

    @property
    def dim_output(self) -> int:
        if self._len_rollouts is None:
            # Only use the last output
            return self.output_layer.out_features
        else:
            return self._len_rollouts * self.output_layer.out_features

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        # See RNNPolicyBase
        if init_values is None:
            # Initialize the layers using default initialization
            init_param(self.rnn_layers, **kwargs)
            init_param(self.output_layer, **kwargs)
        else:
            cp.vector_to_parameters(init_values, self.parameters())

    def summary_statistic(self, data: to.Tensor) -> to.Tensor:
        """
        Pass the time series data through a recurrent neural network.

        .. seealso:
            RNNPolicy with `hidden = None`

        :param data: states and actions of a rollout or segment to be transformed for inference
        :return: features obtained from the RNN at every time step, fattened into a vector
        """
        if self._idcs_data is not None:
            data = data[:, self._idcs_data]

        # Reshape the data to match the shape desired by PyTorch
        data = data.unsqueeze(1)  # shape [len_time_series, 1, dim_data]

        # Pass the input through hidden RNN layers, select the last output, and pass that through the output layer
        out, _ = self.rnn_layers(data, None)

        if self._len_rollouts is None:
            # Only use the output of the last time step
            out = out[-1]

        # Pass through the final output layer
        out = self.output_layer(out)

        # Reshape the outputs of every time step into a 1-dim feature vector
        return out.reshape(-1)


class DynamicTimeWarpingEmbedding(Embedding):
    """
    Embedding for simulation-based inference with time series data which uses the dtw-python package to compute the
    Dynamic Time Warping (DTW) distance between the states as features of the data
    """

    name: str = "dtwemb"
    requires_target_domain_data: bool = True

    def __init__(
        self,
        spec: EnvSpec,
        dim_data: int,
        step_pattern: Optional[str] = None,
        downsampling_factor: int = 1,
        idcs_data: Optional[Union[Tuple[int], List[int]]] = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param spec: environment specification
        :param dim_data: number of dimensions of one data sample, i.e. one time step. By default, this is the sum of the
                         states and action spaces' flat dimensions. This number is doubled if the embedding
                         target domain data.
        :param step_pattern: method passed to dtw-python for computing the distance, e.g. `"symmetric2"` to use
                             dtw-python's default. Here, the default is set to the Rabiner-Juang type VI-c unsmoothed
                             recursion step pattern
        :param downsampling_factor: skip evey `downsampling_factor` time series sample, the downsampling is done in the
                                    base class before calling `summary_statistic()`
        :param idcs_data: list or tuple of integers to select specific states from the data (always using all actions),
                          by default `None` to select all states
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """

        super().__init__(spec, dim_data, downsampling_factor, idcs_data, use_cuda)

        self.step_pattern = step_pattern or rabinerJuangStepPattern(6, "c")
        self.to(self.device)

    @property
    def dim_output(self) -> int:
        return 1

    @to.no_grad()
    def summary_statistic(self, data: to.Tensor) -> to.Tensor:
        """
        Returns the dynamic time warping distance between the simulated rollouts" and the real rollouts' states.

        .. note::
            It is necessary to take the mean over all distances since the same function is used to compute the
            observations (for sbi) form the target domain rollouts. At this point in time there might be only one target
            domain rollout, thus the target domain rollouts are only compared with themselves, thus yield a scalar
            distance value.

        :param data: data tensor containing the simulated states (1st part of the 1st half of the 1st dim) and the
                     real states (1st part of the 2nd half of the 1st dim)
        :return: dynamic time warping distance in multi-dim state space
        """
        # Split the data
        data_sim, data_real = to.chunk(data, 2, dim=1)

        # Extract the states
        if self._idcs_data is not None:
            data_sim = data_sim[:, self._idcs_data]
            data_real = data_real[:, self._idcs_data]
        else:
            data_sim = data_sim[:, : self._env_spec.state_space.flat_dim]
            data_real = data_real[:, : self._env_spec.state_space.flat_dim]

        # Use the dtw package to compute the distance using the specified metric
        data_sim, data_real = data_sim.numpy(), data_real.numpy()
        alignment = dtw(
            data_sim,
            data_real,
            open_end=True,
            step_pattern=self.step_pattern,
        )

        return to.as_tensor(alignment.distance, dtype=to.get_default_dtype()).view(1)
