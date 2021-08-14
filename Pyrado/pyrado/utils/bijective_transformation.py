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
import math
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch as to


class BijectiveTransformation(ABC):
    """
    Base class for bijective transformations to be used for e.g. transforming the domain parameter space of an env.

    These transformations are useful for avoiding infeasible values such as negative masses. The reasoning behind this
    is that some learning methods work on the set of real numbers, thus we make them learn in the transformed space,
    here the log-space, without telling them.
    """

    @abstractmethod
    def forward(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        """
        Map a value from the space to the transformed space.

        :param value: value in the original space
        :return: value in the transformed space
        """
        raise NotImplementedError

    @abstractmethod
    def inverse(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        """
        Map a value from the transformed space to the actual space.

        :param value: value in the transformed space
        :return: value in the original space
        """
        raise NotImplementedError


class LogTransformation(BijectiveTransformation):
    """Transformation to make the values look like they are in log-space."""

    def forward(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            # If value is scalar, i.e., a zero-dimensional value, np.log returns a float instead of an array.
            return np.asarray(np.log(value), dtype=value.dtype)
        elif isinstance(value, to.Tensor):
            return to.log(value)
        else:
            return math.log(value)

    def inverse(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            # If value is scalar, i.e., a zero-dimensional value, np.exp returns a float instead of an array.
            return np.asarray(np.exp(value), dtype=value.dtype)
        elif isinstance(value, to.Tensor):
            return to.exp(value)
        else:
            return math.exp(value)


class SqrtTransformation(BijectiveTransformation):
    """Transformation to make the values look like they are in sqrt-space. This is not actually bijective!"""

    def forward(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            # If value is scalar, i.e., a zero-dimensional value, np.sqrt returns a float instead of an array.
            return np.asarray(np.sqrt(value), dtype=value.dtype)
        elif isinstance(value, to.Tensor):
            return to.sqrt(value)
        else:
            return math.sqrt(value)

    def inverse(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            # If value is scalar, i.e., a zero-dimensional value, np.power returns a float instead of an array.
            return np.asarray(np.power(value, 2), dtype=value.dtype)
        elif isinstance(value, to.Tensor):
            return to.pow(value, 2)
        else:
            return math.pow(value, 2)


class IdentityTransformation(BijectiveTransformation):
    """Transformation that does nothing."""

    def forward(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        return value

    def inverse(self, value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        return value
