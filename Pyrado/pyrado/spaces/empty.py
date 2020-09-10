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

import numpy as np
from tabulate import tabulate
from typing import Sequence, Union

import pyrado
from pyrado.spaces.base import Space
from pyrado.utils.input_output import color_validity


class EmptySpace(Space):
    """ A space with no content """

    def _members(self) -> tuple:
        # We're a singleton, compare by id
        return id(self),

    @property
    def bounds(self) -> tuple:
        return np.array([]), np.array([])

    @property
    def labels(self) -> (np.ndarray, None):
        return np.array([], dtype=np.object)

    @property
    def shape(self) -> tuple:
        return (1,)  # (0,) would be better, but that causes the param init function form PyTorch to crash

    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        raise NotImplementedError("Cannot shrink empty space!")

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        # Check the candidate
        if not cand.shape == self.shape:
            raise pyrado.ShapeErr(given=cand, expected_match=self)
        if np.isnan(cand).any():
            raise pyrado.ValueErr(
                msg=f'At least one value is NaN!' +
                    tabulate([list(self.labels), [*color_validity(cand, np.invert(np.isnan(cand)))]],
                             headers='firstrow')
            )
        return True

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        return np.array([])

    def project_to(self, ele: np.ndarray) -> np.ndarray:
        return np.array([])

    def subspace(self, idcs: Union[np.ndarray, int, slice] = None):
        return self

    @staticmethod
    def cat(spaces: Union[list, tuple]):
        if not all(isinstance(s, EmptySpace) for s in spaces):
            raise pyrado.TypeErr(given=spaces, expected_type=Sequence[EmptySpace])
        return EmptySpace
