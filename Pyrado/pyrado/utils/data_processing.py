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
Only tested for 1-dim inputs, e.g. time series of rewards.
"""
import numpy as np
import torch as to
from typing import Union, Tuple

import pyrado


def scale_min_max(
    data: Union[np.ndarray, to.Tensor],
    bound_lo: Union[int, float, np.ndarray, to.Tensor],
    bound_up: Union[int, float, np.ndarray, to.Tensor],
) -> Union[np.ndarray, to.Tensor]:
    r"""
    Transform the input data to to be in $[a, b]$.

    :param data: unscaled input ndarray or Tensor
    :param bound_lo: lower bound for the transformed data
    :param bound_up: upper bound for the transformed data
    :return: ndarray or Tensor scaled to be in $[a, b]$
    """
    # Lower bound
    if isinstance(bound_lo, (float, int)) and isinstance(data, np.ndarray):
        bound_lo = bound_lo * np.ones_like(data, dtype=np.float64)
    elif isinstance(bound_lo, (float, int)) and isinstance(data, to.Tensor):
        bound_lo = bound_lo * to.ones_like(data, dtype=to.get_default_dtype())
    elif isinstance(bound_lo, np.ndarray) and isinstance(data, to.Tensor):
        bound_lo = to.from_numpy(bound_lo)
    elif isinstance(bound_lo, to.Tensor) and isinstance(data, np.ndarray):
        bound_lo = bound_lo.numpy()

    # Upper bound
    if isinstance(bound_up, (float, int)) and isinstance(data, np.ndarray):
        bound_up = bound_up * np.ones_like(data, dtype=np.float64)
    elif isinstance(bound_up, (float, int)) and isinstance(data, to.Tensor):
        bound_up = bound_up * to.ones_like(data, dtype=to.get_default_dtype())
    elif isinstance(bound_up, np.ndarray) and isinstance(data, to.Tensor):
        bound_up = to.from_numpy(bound_up)
    elif isinstance(bound_up, to.Tensor) and isinstance(data, np.ndarray):
        bound_up = bound_up.numpy()

    if not (bound_lo < bound_up).all():
        raise pyrado.ValueErr(given_name="lower bound", l_constraint="upper bound")

    if isinstance(data, np.ndarray):
        data_ = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif isinstance(data, to.Tensor):
        data_ = (data - to.min(data)) / (to.max(data) - to.min(data))
    else:
        raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

    return data_ * (bound_up - bound_lo) + bound_lo


def standardize(data: Union[np.ndarray, to.Tensor], eps: float = 1e-8) -> Union[np.ndarray, to.Tensor]:
    r"""
    Standardize the input data to make it $~ N(0, 1)$.

    :param data: input ndarray or Tensor
    :param eps: factor for numerical stability
    :return: standardized ndarray or Tensor
    """
    if isinstance(data, np.ndarray):
        return (data - np.mean(data)) / (np.std(data) + float(eps))
    elif isinstance(data, to.Tensor):
        return (data - to.mean(data)) / (to.std(data) + float(eps))
    else:
        raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])


def normalize(
    x: Union[np.ndarray, to.Tensor], axis: int = -1, order: int = 1, eps: float = 1e-8
) -> Union[np.ndarray, to.Tensor]:
    """
    Normalize a numpy `ndarray` or a PyTroch `Tensor` without changing the input.
    Choosing `axis=1` and `norm_order=1` makes all columns of sum to 1.

    :param x: input to normalize
    :param axis: axis of the array to normalize along
    :param order: order of the norm (e.g., L1 norm: absolute values, L2 norm: quadratic values)
    :param eps: lower bound on the norm, to avoid division by zero
    :return: normalized array
    """
    if isinstance(x, np.ndarray):
        norm_x = np.atleast_1d(np.linalg.norm(x, ord=order, axis=axis))  # calculate norm over axis
        norm_x = np.where(norm_x > eps, norm_x, np.ones_like(norm_x))  # avoid division by 0
        return x / np.expand_dims(norm_x, axis)  # element wise division
    elif isinstance(x, to.Tensor):
        norm_x = to.norm(x, p=order, dim=axis)  # calculate norm over axis
        norm_x = to.where(norm_x > eps, norm_x, to.ones_like(norm_x))  # avoid division by 0
        return x / norm_x.unsqueeze(axis)  # element wise division
    else:
        raise pyrado.TypeErr(given=x, expected_type=[np.array, to.Tensor])


class Standardizer:
    """ A stateful standardizer that remembers the mean and standard deviation for later un-standardization """

    def __init__(self):
        self.mean = None
        self.std = None

    def standardize(self, data: Union[np.ndarray, to.Tensor], eps: float = 1e-8) -> Union[np.ndarray, to.Tensor]:
        r"""
        Standardize the input data to make it $~ N(0, 1)$ and store the input's mean and standard deviation.

        :param data: input ndarray or Tensor
        :param eps: factor for numerical stability
        :return: standardized ndarray or Tensor
        """
        if isinstance(data, np.ndarray):
            self.mean = np.mean(data)
            self.std = np.std(data)
            return (data - self.mean) / (self.std + float(eps))
        elif isinstance(data, to.Tensor):
            self.mean = to.mean(data)
            self.std = to.std(data)
            return (data - self.mean) / (self.std + float(eps))
        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

    def unstandardize(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        r"""
        Revert the previous standardization of the input data to make it $~ N(\mu, \sigma)$.

        :param data: input ndarray or Tensor
        :return: un-standardized ndarray or Tensor
        """
        if self.mean is None or self.std is None:
            raise pyrado.ValueErr(msg="Use standardize before unstandardize!")

        # Input type must match stored type
        if isinstance(data, np.ndarray) and isinstance(self.mean, np.ndarray):
            pass
        elif isinstance(data, to.Tensor) and isinstance(self.mean, to.Tensor):
            pass
        elif isinstance(data, np.ndarray) and isinstance(self.mean, to.Tensor):
            self.mean = self.mean.numpy()
            self.std = self.std.numpy()
        elif isinstance(data, to.Tensor) and isinstance(self.mean, np.ndarray):
            self.mean = to.from_numpy(self.mean)
            self.std = to.from_numpy(self.std)

        x_unstd = data * self.std + self.mean
        return x_unstd


class MinMaxScaler:
    """ A stateful min-max scaler that remembers the lower and upper bound for later un-unscaling """

    def __init__(
        self, bound_lo: Union[int, float, np.ndarray, to.Tensor], bound_up: Union[int, float, np.ndarray, to.Tensor]
    ):
        """
        Constructor

        :param bound_lo: lower bound for the transformed data
        :param bound_up: upper bound for the transformed data
        """
        # Store the values as private members since they are not directly used
        self._bound_lo = bound_lo
        self._bound_up = bound_up

        # Initialize in scale_to()
        self.data_min = None
        self.data_span = None

    def _convert_bounds(
        self, data: Union[to.Tensor, np.ndarray]
    ) -> Union[Tuple[to.Tensor, to.Tensor], Tuple[np.ndarray, np.ndarray]]:
        """
        Convert the bounds into the right type

        :param data: data that is later used for projecting
        :return: bounds casted to the type of data
        """
        # Lower bound
        if isinstance(self._bound_lo, (float, int)) and isinstance(data, np.ndarray):
            bound_lo = self._bound_lo * np.ones_like(data, dtype=np.float64)
        elif isinstance(self._bound_lo, (float, int)) and isinstance(data, to.Tensor):
            bound_lo = self._bound_lo * to.ones_like(data, dtype=to.get_default_dtype())
        elif isinstance(self._bound_lo, np.ndarray) and isinstance(data, to.Tensor):
            bound_lo = to.from_numpy(self._bound_lo)
        elif isinstance(self._bound_lo, to.Tensor) and isinstance(data, np.ndarray):
            bound_lo = self._bound_lo.numpy()
        else:
            bound_lo = self._bound_lo

        # Upper bound
        if isinstance(self._bound_up, (float, int)) and isinstance(data, np.ndarray):
            bound_up = self._bound_up * np.ones_like(data, dtype=np.float64)
        elif isinstance(self._bound_up, (float, int)) and isinstance(data, to.Tensor):
            bound_up = self._bound_up * to.ones_like(data, dtype=to.get_default_dtype())
        elif isinstance(self._bound_up, np.ndarray) and isinstance(data, to.Tensor):
            bound_up = to.from_numpy(self._bound_up)
        elif isinstance(self._bound_up, to.Tensor) and isinstance(data, np.ndarray):
            bound_up = self._bound_up.numpy()
        else:
            bound_up = self._bound_up

        return bound_lo, bound_up

    def scale_to(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        r"""
        Transform the input data to be in $[a, b]$, where $a$ and $b$ are defined during construction.

        :param data: unscaled input ndarray or Tensor
        :return: ndarray or Tensor scaled to be in $[a, b]$
        """
        # Convert to the right type if necessary
        bound_lo, bound_up = self._convert_bounds(data)

        if not (bound_lo < bound_up).all():
            raise pyrado.ValueErr(given_name="lower bound", l_constraint="upper bound")

        if isinstance(data, np.ndarray):
            self._data_min = np.min(data)
            self._data_span = np.max(data) - np.min(data)
        elif isinstance(data, to.Tensor):
            self._data_min = to.min(data)
            self._data_span = to.max(data) - to.min(data)
        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

        data_ = (data - self._data_min) / self._data_span
        return data_ * (bound_up - bound_lo) + bound_lo

    def scale_back(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        r"""
        Rescale the input data back to its original value range

        :param data: input ndarray or Tensor scaled to be in $[a, b]$
        :return: unscaled ndarray or Tensor
        """
        if self._data_min is None or self._data_span is None:
            raise pyrado.ValueErr(msg="Call scale_to before scale_back!")

        if not (isinstance(data, np.ndarray) or isinstance(data, to.Tensor)):
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

        # Convert to the right type if necessary
        bound_lo, bound_up = self._convert_bounds(data)

        # Scale data to be in [0, 1]
        data_ = (data - bound_lo) / (bound_up - bound_lo)

        # Return data in original value range
        return data_ * self._data_span + self._data_min


class RunningStandardizer:
    """ Implementation of Welford's online algorithm """

    def __init__(self):
        """ Constructor """
        self.mean = None
        self.sum_sq_diffs = None  # a.k.a M2
        self.iter = 0

    def reset(self):
        """ Reset internal variables. """
        self.mean = None
        self.sum_sq_diffs = None
        self.iter = 0

    def __repr__(self):
        return (
            f"RunningStandardizer ID: {id(self)}\n"
            f"mean: {self.mean}\nss_diffs: {self.sum_sq_diffs}\niter: {self.iter}"
        )

    def __call__(self, data: Union[np.ndarray, to.Tensor], axis: int = 0):
        """
        Update the internal variables and standardize the input.

        :param data: input data to be standardized
        :param axis: axis to standardized along
        :return: standardized data
        """
        if isinstance(data, np.ndarray):
            # Process element wise (keeps dim) or average along one axis
            mean = np.mean(data, axis=axis)

            self.iter += 1

            # Handle first iteration separately
            if self.iter <= 1:
                # delta = x
                self.mean = mean
                self.sum_sq_diffs = mean * mean
                return data
            else:
                # Update mean
                delta_prev = mean - self.mean
                self.mean += delta_prev / self.iter

                # Update sum of squares of differences from the current mean
                delta_curr = mean - self.mean
                self.sum_sq_diffs += delta_prev * delta_curr

                # Calculate the unbiased sample variance
                var_sample = self.sum_sq_diffs / (self.iter - 1)

                # Return normalized data
                return (data - self.mean) / np.sqrt(var_sample)

        elif isinstance(data, to.Tensor):
            # Process element wise (keeps dim) or average along one axis
            mean = to.mean(data, dim=axis).to(to.get_default_dtype())

            self.iter += 1

            # Handle first iteration separately
            if self.iter <= 1:
                # delta = x
                self.mean = mean
                self.sum_sq_diffs = mean * mean
                return data
            else:
                # Update mean
                delta_prev = mean - self.mean
                self.mean += delta_prev / self.iter

                # Update sum of squares of differences from the current mean
                delta_curr = mean - self.mean
                self.sum_sq_diffs += delta_prev * delta_curr

                # Calculate the unbiased sample variance
                var_sample = self.sum_sq_diffs / (self.iter - 1)

                # Return normalized data
                return (data - self.mean) / to.sqrt(var_sample)

        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])


class RunningNormalizer:
    """ Normalizes given data based on the history of observed data, such that all outputs are in range [-1, 1] """

    def __init__(self):
        """ Constructor """
        self.bound_lo = None
        self.bound_up = None
        self.eps = 1e-3
        self.iter = 0

    def reset(self):
        """ Reset internal variables. """
        self.bound_lo = None
        self.bound_up = None
        self.iter = 0

    def __repr__(self):
        return (
            f"RunningNormalizer ID: {id(self)}\n"
            f"bound_lo: {self.bound_lo}\nbound_up: {self.bound_up}\niter: {self.iter}"
        )

    def __call__(self, data: Union[np.ndarray, to.Tensor]):
        """
        Update the internal variables and normalize the input.

        :param data: input data to be standardized
        :return: normalized data in [-1, 1]
        """
        if isinstance(data, np.ndarray):
            data_2d = np.atleast_2d(data)
            data_min = np.min(data_2d, axis=0)
            data_max = np.max(data_2d, axis=0)
            self.iter += 1

            # Handle first iteration separately
            if self.iter <= 1:
                self.bound_lo = data_min
                self.bound_up = data_max
            else:
                if not self.bound_lo.shape == data_min.shape:
                    raise pyrado.ShapeErr(given=data_min, expected_match=self.bound_lo)

                # Update bounds with element wise
                self.bound_lo = np.fmin(self.bound_lo, data_min)
                self.bound_up = np.fmax(self.bound_up, data_max)

            # Make sure that the bounds do not collapse (e.g. for one sample)
            if np.linalg.norm(self.bound_up - self.bound_lo, ord=1) < self.eps:
                self.bound_lo -= self.eps / 2
                self.bound_up += self.eps / 2

        elif isinstance(data, to.Tensor):
            data_2d = data.view(-1, 1) if data.ndim < 2 else data
            data_min, _ = to.min(data_2d, dim=0)
            data_max, _ = to.max(data_2d, dim=0)
            self.iter += 1

            # Handle first iteration separately
            if self.iter <= 1:
                self.bound_lo = data_min
                self.bound_up = data_max
            else:
                if not self.bound_lo.shape == data_min.shape:
                    raise pyrado.ShapeErr(given=data_min, expected_match=self.bound_lo)

                # Update bounds with element wise
                self.bound_lo = to.min(self.bound_lo, data_min)
                self.bound_up = to.max(self.bound_up, data_max)

            # Make sure that the bounds do not collapse (e.g. for one sample)
            if to.norm(self.bound_up - self.bound_lo, p=1) < self.eps:
                self.bound_lo -= self.eps / 2
                self.bound_up += self.eps / 2

        else:
            raise pyrado.TypeErr(given=data, expected_type=[np.ndarray, to.Tensor])

        # Return standardized data
        return (data - self.bound_lo) / (self.bound_up - self.bound_lo) * 2 - 1
