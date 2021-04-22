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

"""
Only tested for 1-dim inputs.
"""
import numpy as np
import torch as to

import pyrado


class RunningExpDecayingAverage:
    """ Implementation of an exponentially decaying average estimator """

    def __init__(self, alpha: float):
        """
        Constructor

        :param alpha: decay factor
        """
        if not 0 <= alpha <= 1.0:
            raise pyrado.ValueErr(given=alpha, ge_constraint="0", le_constraint="1")
        self._prev_est = None  # previous estimate of the mean
        self._alpha = alpha

    def reset(self, alpha: float = None):
        """ Reset internal variables. """
        if alpha is not None:
            if not 0 <= alpha <= 1.0:
                raise pyrado.ValueErr(given=alpha, ge_constraint="0", le_constraint="1")
            self._alpha = alpha
        self._prev_est = None  # leads to re-init on call

    def __repr__(self):
        return f"RunningExpDecayingAverage ID: {id(self)}\n" f"mean: {self._prev_est}\nalpha: {self._alpha}"

    def __call__(self, data: [np.ndarray, to.Tensor]) -> [np.ndarray, to.Tensor]:
        """
        Update the internal variables and compute the exponentially decayed running average.

        :param data: input data
        :return: estimated average
        """
        if isinstance(data, np.ndarray):
            if self._prev_est is None:
                self._prev_est = data.copy()
            else:
                # Store current estimate in the _prev_est attribute for the next iteration
                self._prev_est = self._alpha * data + (1.0 - self._alpha) * self._prev_est
            # Return current estimate
            return self._prev_est.copy()

        elif isinstance(data, to.Tensor):
            if self._prev_est is None:
                self._prev_est = data.clone()
            else:
                # Store current estimate in the _prev_est attribute for the next iteration
                self._prev_est = self._alpha * data + (1.0 - self._alpha) * self._prev_est
            # Return current estimate
            return self._prev_est.clone()

        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])


class RunningMemoryAverage:
    """ Implementation of an estimator that computes the average for a memorized buffer """

    def __init__(self, capacity: int):
        """
        Constructor

        :param capacity: memory size
        """
        if not 1 <= capacity:
            raise pyrado.ValueErr(given=capacity, ge_constraint="1")
        self.capacity = capacity
        self._memory = None

    @property
    def memory(self) -> [np.ndarray, to.Tensor, None]:
        return self._memory

    def reset(self, capacity: float = None):
        """ Reset internal variables. """
        if capacity is not None:
            if not 1 <= capacity:
                raise pyrado.ValueErr(given=capacity, ge_constraint="1")
        self.capacity = capacity
        self._memory = None

    def __call__(self, data: [np.ndarray, to.Tensor]) -> [np.ndarray, to.Tensor]:
        """
        Update the internal variables and compute the exponentially decayed running average.

        :param data: input data
        :return: unweighted average
        """
        if data.ndim > 1:
            raise pyrado.ShapeErr(msg="RunningMemoryAverage only supports scalars and vectors")

        if isinstance(data, np.ndarray):
            # Format new data
            new = data.copy().reshape(1, data.size)

            if self._memory is None:
                self._memory = new
            else:
                self._memory = np.concatenate([self._memory, new], axis=0)

            # Drop surplus
            self._memory = self._memory[-self.capacity :]  # pylint: disable=invalid-unary-operand-type

            # Return current estimate
            return np.mean(self._memory, axis=0)

        elif isinstance(data, to.Tensor):
            # Format new data
            new = data.clone().view(1, to.prod(to.tensor(data.size())))

            if self._memory is None:
                self._memory = new
            else:
                self._memory = to.cat([self._memory, new], dim=0)

            # Drop surplus
            self._memory = self._memory[-self.capacity :]  # pylint: disable=invalid-unary-operand-type

            # Return current estimate
            return to.mean(self._memory, dim=0)

        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])
