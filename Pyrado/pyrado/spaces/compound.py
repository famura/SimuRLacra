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

from copy import deepcopy
from typing import Sequence, Union

import numpy as np

from pyrado.spaces.base import Space
from pyrado.utils.input_output import print_cbt


class CompoundSpace(Space):
    """ Space consisting of other spaces """

    def __init__(self, spaces: Sequence[Space]):
        """
        Constructor

        :param spaces: list or tuple of spaces to sample from randomly
        """
        self._spaces = deepcopy(spaces)

    @property
    def shape(self):
        # Return all shapes
        return (s for s in self._spaces)

    @property
    def flat_dim(self) -> int:
        return sum([s.flat_dim for s in self._spaces])

    def _members(self):
        # Return the subspaces
        return self._spaces

    def project_to(self, ele: np.ndarray):
        raise NotImplementedError

    def subspace(self, idcs: Union[int, slice]):
        # Subspace of this CompoundSpace and not of the individual spaces
        return self._spaces[idcs]

    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        raise NotImplementedError

    @staticmethod
    def cat(spaces: Union[list, tuple]):
        raise NotImplementedError

    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        valid = any([s.contains(cand) for s in self._spaces])
        if not valid and verbose:
            print_cbt(f"Violated all of the {len(self._spaces)} subspaces!", "r")
            for s in self._spaces:
                s.contains(cand, verbose)
        return valid

    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        # Sample a subspace and then sample from this subspace
        idx = np.random.randint(len(self._spaces))
        return self._spaces[idx].sample_uniform()
