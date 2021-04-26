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
from collections.abc import Iterable
from copy import deepcopy
from functools import reduce
from typing import Tuple, Union

import numpy as np
from colorama import Style
from tabulate import tabulate

import pyrado
from pyrado.utils import get_class_name


class Space(ABC):
    """Base class of all state, action, and init spaces in Pyrado"""

    bound_lo: np.ndarray
    bound_up: np.ndarray

    def __str__(self):
        """Get an information string."""
        return (
            f"{Style.BRIGHT}{get_class_name(self)}{Style.RESET_ALL}"
            + f"(id: {id(self)})\n"
            + tabulate(
                [[b for b in self.bound_lo], [b for b in self.bound_up]],
                headers=[*self.labels],
                showindex=["lower", "upper"],
            )
        )

    @property
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the lower (first element) and upper bound (second element) of the space."""
        return self.bound_lo, self.bound_up

    @property
    def bound_abs_up(self) -> np.ndarray:
        """Get the upper bound in terms of absolute values."""
        return np.max(np.stack([np.abs(self.bound_lo), np.abs(self.bound_up)], axis=0), axis=0)

    @property
    def ele_dim(self) -> int:
        """Get the dimension of an element of the space."""
        return self.bound_lo.shape[0]

    @property
    @abstractmethod
    def shape(self) -> tuple:
        """Get the shape of the full space."""
        raise NotImplementedError

    @property
    def flat_dim(self) -> int:
        """Get the dimension of the flattened space."""
        return reduce((lambda x, y: x * y), self.shape)

    @property
    def labels(self) -> Union[np.ndarray, None]:
        """Get the labels for space entries, or `None` if not supported."""
        return None

    @abstractmethod
    def _members(self) -> tuple:
        """Return a tuple of members relevant for equals."""
        raise NotImplementedError

    def __eq__(self, other):
        if type(other) is type(self):
            return np.all([s == o for s, o in zip(self._members(), other._members())])
        else:
            return False

    def create_mask(self, *idcs):
        """
        Create a mask selecting the given indices from this space.
        Every index should be a number or a name in labels.

        :param idcs: index list, which can either be varargs or a single iterable
        :return: mask array with 1 at each index
        """
        mask = np.zeros(self.shape, dtype=np.bool_)

        if len(idcs) == 1 and isinstance(idcs[0], Iterable) and not isinstance(idcs[0], str):
            # Unwrap single iterable argument
            idcs = idcs[0]

        labels = self.labels
        # Set selected values to 1
        for idx in idcs:
            if isinstance(idx, str):
                # Handle labels
                assert labels is not None, "The space must be labeled to use label-based indexing"
                for idx_label, label in np.ndenumerate(labels):
                    if label == idx:
                        idx = idx_label
                        break
                else:
                    raise pyrado.ValueErr(msg=f"Label {idx} not found in {self}")
            if np.all(mask[idx] == 1):
                label_desc = f" ({labels[idx]})" if labels is not None else ""
                raise pyrado.ValueErr(msg=f"Duplicate index {idx}{label_desc}")
            mask[idx] = 1

        return mask

    def copy(self):
        """Create a deep copy (recursively copy values of a compound object)."""
        return deepcopy(self)

    def subspace(self, idcs: Union[np.ndarray, int, slice]):
        """
        Select a subspace by passing an array or a list of indices. The oder is preserved.

        :param idcs: indices or mask, entries with `True` are kept
        :return: subspace with the same boundaries but reduced dimensionality
        """
        raise NotImplementedError

    @abstractmethod
    def shrink(self, new_lo: np.ndarray, new_up: np.ndarray):
        """
        Create a convex subspace of the existing space.

        :param new_lo: array containing the new lower bound
        :param new_up: array containing the new upper bound
        :return: subspace that is bounded by the original space
        """
        raise NotImplementedError

    @abstractmethod
    def contains(self, cand: np.ndarray, verbose: bool = False) -> bool:
        """
        Check if a candidate element is in the state.

        :param cand: candidate to check if it is an element of the space
        :param verbose: if `True` details are be printed in case the candidate is not an element of the space
        :return: bool value
        """
        raise NotImplementedError

    def __contains__(self, cand):
        # Delegate to actual method, which may be overridden
        return self.contains(cand)

    @abstractmethod
    def sample_uniform(self, concrete_inf: float = 1e6) -> np.ndarray:
        """
        Sample an element of this space using a uniform distribution.

        :param concrete_inf: It's impossible to sample uniform from infinite bounds. As a workaround, infinite
                             bounds are replaced with this value.
        :return: element within the space
        """
        raise NotImplementedError

    @abstractmethod
    def project_to(self, ele: np.ndarray) -> np.ndarray:
        """
        Project an into the box space by clipping, or do nothing of it already is.

        :param ele: element from outside or inside of the box space
        :return: element within the box space (at the border if it has been outside of the space before the projection).
        """
        raise NotImplementedError

    @staticmethod
    def cat(spaces: Union[list, tuple]):
        """
        Concatenate a sequence of spaces in the given order. This creates a new Space object.

        :param spaces: spaces to concatenate
        :return: concatenated space
        """
        raise NotImplementedError
