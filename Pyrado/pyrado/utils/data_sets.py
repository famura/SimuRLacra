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

import random
import torch as to
from tabulate import tabulate
from torch.utils.data.dataset import Dataset
from typing import List, Tuple, Union

import pyrado
from pyrado.utils.data_processing import standardize, scale_min_max
from pyrado.utils.data_types import TimeSeriesDataPair
from pyrado.utils.input_output import print_cbt


def create_sequences(data: to.Tensor, len_seq: int) -> List[Tuple]:
    """
    Create input sequences with associated target values from data. The targets are the next data point after an input
    sequence of given length.

    :param data: data to be split into inputs and targets
    :param len_seq: length of the input sequences
    :return: list of tuples with inout and target data pairs
    """
    list_inp_targ = []
    for idx in range(data.shape[0] - len_seq - 1):
        inp = data[idx : (idx + len_seq)]
        targ = data[idx + len_seq]
        list_inp_targ.append(TimeSeriesDataPair(inp, targ))
    return list_inp_targ


def create_shuffled_sequences(data: to.Tensor, len_seq: int) -> List[Tuple]:
    """
    Create  input sequences with associated target values from data using `create_sequences()`, and shuffle the order
    of the sequences afterwards.

    :param data: data to be split into inputs and targets
    :param len_seq: length of the input sequences
    :return: list of randomly ordered tuples with inout and target data pairs
    """
    # Shuffle inputs and targets the same way
    seqs_inp_targ = create_sequences(data, len_seq)
    seqs_inp_targ_rand = random.sample(seqs_inp_targ, len(seqs_inp_targ))
    return seqs_inp_targ_rand


class TimeSeriesDataSet(Dataset):
    """ Class for storing time series data sets """

    def __init__(
        self,
        data: to.Tensor,
        window_size: int,
        ratio_train: float,
        standardize_data: bool = False,
        scale_min_max_data: bool = False,
        name: str = "Unnamed data set",
    ):
        r"""
        Constructor

        :param data: complete raw data set, where the samples are along the first dimension
        :param window_size: length of the sequences fed to the policy for predicting the next value
        :param ratio_train: ratio of the training samples w.r.t. the total sample count
        :param standardize_data: if `True`, the data is standardized to be $~ N(0,1)$
        :param scale_min_max_data:  if `True`, the data is scaled to $\in [-1, 1]$
        :param name: descriptive name for the data set
        """
        if not isinstance(data, to.Tensor):
            raise pyrado.TypeErr(given=data, expected_type=to.Tensor)
        if not isinstance(window_size, int):
            raise pyrado.TypeErr(given=window_size, expected_type=int)
        if window_size < 1:
            raise pyrado.ValueErr(given=window_size, ge_constraint="1")
        if not isinstance(ratio_train, float):
            raise pyrado.TypeErr(given=ratio_train, expected_type=float)
        if not (0 < ratio_train < 1):
            raise pyrado.ValueErr(given=ratio_train, g_constraint="0", l_constraint="1")
        if standardize_data and scale_min_max_data:
            raise pyrado.ValueErr(msg="Scaling and normalizing the data at the same time is not supported!")

        self.data_all_raw = to.atleast_2d(data).T if data.ndimension() == 1 else data  # samples along rows
        self._ratio_train = ratio_train
        self._window_size = window_size
        self.name = name

        # Process the data
        self.is_standardized, self.is_scaled = False, False
        if standardize_data:
            self.data_all = standardize(self.data_all_raw)  # ~ N(0,1)
            self.is_standardized = True
        elif scale_min_max_data:
            self.data_all = scale_min_max(self.data_all_raw, -1, 1)  # in [-1, 1]
            self.is_scaled = True
        else:
            self.data_all = self.data_all_raw

        # Split the data into training and testing data
        self.data_trn = self.data_all[: self.num_samples_trn]
        self.data_tst = self.data_all[self.num_samples_trn :]

        # Targets are the next time steps
        self.data_all_inp = self.data_all[:-1, :]
        self.data_trn_inp = self.data_trn[:-1, :]
        self.data_tst_inp = self.data_tst[:-1, :]
        self.data_all_targ = self.data_all[1:, :]
        self.data_trn_targ = self.data_trn[1:, :]
        self.data_tst_targ = self.data_tst[1:, :]

        # Create sequences
        self.data_trn_ws = self.cut_to_window_size(self.data_trn, self._window_size)
        self.data_tst_ws = self.cut_to_window_size(self.data_tst, self._window_size)
        self.data_trn_seqs = create_sequences(self.data_trn_ws, len_seq=self._window_size + 1)
        self.data_tst_seqs = create_sequences(self.data_tst_ws, len_seq=self._window_size + 1)

        print_cbt(f"Created {str(self)}", "w")

    def __len__(self) -> int:
        """ Get the length of the complete data set (not split into sequences) after removing superfluous samples. """
        return self.data_all.shape[0]

    def __getitem__(self, idx: Union[int, slice]):
        """ Get one sample from the complete data set (not split into sequences). """
        return self.data_all[idx, :]

    def __eq__(self, other) -> bool:
        """ Check if two data sets are equal by comparing the data and properties. """
        data_eq = to.allclose(self.data_all, other.data_all)
        ratio_eq = self.ratio_train == other.ratio_train
        window_eq = self.window_size == other.window_size
        return data_eq and ratio_eq and window_eq

    def __str__(self):
        """ Get an information string. """
        return f"TimeSeriesDataSet (id {id(self)})\n" + tabulate(
            [
                ["num all samples", len(self)],
                ["ratio trn samples", self.ratio_train],
                ["num training sequences", len(self.data_trn_seqs)],
                ["window size", self.window_size],
            ]
        )

    @property
    def ratio_train(self) -> float:
        """ Get the ratio of the training samples w.r.t. the total sample count. """
        return self._ratio_train

    @property
    def window_size(self) -> int:
        """ Get the length of the sequences fed to the policy for predicting the next value. """
        return self._window_size

    @property
    def num_samples_trn(self) -> int:
        """ Get the number of samples in the training subset. """
        return int(len(self) * self._ratio_train)

    @property
    def num_samples_tst(self) -> int:
        """ Get the number of samples in the testing subset. """
        return len(self) - self.num_samples_trn

    @staticmethod
    def cut_to_window_size(data: to.Tensor, ws: int) -> to.Tensor:
        """
        Cut off samples such that all training sequences have length `window_size`.

        :param data: input data set with samples along the first dimension
        :param ws: input window length as used for learning
        :return: data of proper length
        """
        if not isinstance(ws, int):
            raise pyrado.TypeErr(given=ws, expected_type=int)

        num_cutoff_samples = data.shape[0] % (ws + 1)
        if num_cutoff_samples > 0:
            data = data[:-num_cutoff_samples]
        return data
