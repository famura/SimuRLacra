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
import torch as to

import pyrado


class RunningNormalizer:
    """ Normalizes given data based on the history of observed data, such that all outputs are in range [-1, 1] """

    def __init__(self):
        """ Constructor """
        self._bound_lo = None
        self._bound_up = None
        self.eps = 1e-3
        self._iter = 0

    def reset(self):
        """ Reset internal variables. """
        self._bound_lo = None
        self._bound_up = None
        self._iter = 0

    def __repr__(self):
        return f'RunningNormalizer ID: {id(self)}\n' \
               f'bound_lo: {self._bound_lo}\nbound_up: {self._bound_up}\niter: {self._iter}'

    def __call__(self, data: [np.ndarray, to.Tensor]):
        """
        Update the internal variables and normalize the input.

        :param data: input data to be standardized
        :return: normalized data in [-1, 1]
        """
        if isinstance(data, np.ndarray):
            data_2d = np.atleast_2d(data)
            data_min = np.min(data_2d, axis=0)
            data_max = np.max(data_2d, axis=0)
            self._iter += 1

            # Handle first iteration separately
            if self._iter <= 1:
                self._bound_lo = data_min
                self._bound_up = data_max
            else:
                if not self._bound_lo.shape == data_min.shape:
                    raise pyrado.ShapeErr(given=data_min, expected_match=self._bound_lo)

                # Update bounds with element wise
                self._bound_lo = np.fmin(self._bound_lo, data_min)
                self._bound_up = np.fmax(self._bound_up, data_max)

            # Make sure that the bounds do not collapse (e.g. for one sample)
            if np.linalg.norm(self._bound_up - self._bound_lo, ord=1) < self.eps:
                self._bound_lo -= self.eps/2
                self._bound_up += self.eps/2

        elif isinstance(data, to.Tensor):
            data_2d = data.view(-1, 1) if data.ndim < 2 else data
            data_min, _ = to.min(data_2d, dim=0)
            data_max, _ = to.max(data_2d, dim=0)
            self._iter += 1

            # Handle first iteration separately
            if self._iter <= 1:
                self._bound_lo = data_min
                self._bound_up = data_max
            else:
                if not self._bound_lo.shape == data_min.shape:
                    raise pyrado.ShapeErr(given=data_min, expected_match=self._bound_lo)

                # Update bounds with element wise
                self._bound_lo = to.min(self._bound_lo, data_min)
                self._bound_up = to.max(self._bound_up, data_max)

            # Make sure that the bounds do not collapse (e.g. for one sample)
            if to.norm(self._bound_up - self._bound_lo, p=1) < self.eps:
                self._bound_lo -= self.eps/2
                self._bound_up += self.eps/2

        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

        # Return standardized data
        return (data - self._bound_lo)/(self._bound_up - self._bound_lo)*2 - 1


def normalize(x: [np.ndarray, to.Tensor], axis: int = -1, order: int = 1, eps: float = 1e-8) -> (np.ndarray, to.Tensor):
    """
    Normalize a numpy `ndarray` or a PyTroch `Tensor` without changing the input.
    Choosing `axis=1` and `norm_order=1` makes all columns of sum to 1.

    :param x: input to normalize
    :param axis: axis of the array to normalize along
    :param order: order of the norm (e.g., L_1 norm: absolute values, L_2 norm: quadratic values)
    :param eps: lower bound on the norm, to avoid division by zero
    :return: normalized array
    """
    if isinstance(x, np.ndarray):
        norm_x = np.atleast_1d(np.linalg.norm(x, ord=order, axis=axis))  # calculate norm over axis
        norm_x = np.where(norm_x > eps, norm_x, np.ones_like(norm_x))  # avoid division by 0
        return x/np.expand_dims(norm_x, axis)  # element wise division
    elif isinstance(x, to.Tensor):
        norm_x = to.norm(x, p=order, dim=axis)  # calculate norm over axis
        norm_x = to.where(norm_x > eps, norm_x, to.ones_like(norm_x))  # avoid division by 0
        return x/norm_x.unsqueeze(axis)  # element wise division
    else:
        raise pyrado.TypeErr(given=x, expected_type=[np.array, to.Tensor])
