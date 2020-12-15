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
from scipy.special import logsumexp
from typing import Union, Tuple

import pyrado
from pyrado.utils.data_processing import normalize


class UnitCubeProjector:
    """ Project to a unit qube $[0, 1]^d$ and back using explicit bounds """

    def __init__(self, bound_lo: Union[np.ndarray, to.Tensor], bound_up: Union[np.ndarray, to.Tensor]):
        """
        Constructor

        :param bound_lo: lower bound
        :param bound_up: upper bound
        """
        if not type(bound_lo) == type(bound_up):
            raise pyrado.TypeErr(msg="Passed two different types for bounds!")
        if any(bound_lo == pyrado.inf):
            raise pyrado.ValueErr(given=bound_lo, eq_constraint="not +/- inf")
        if any(bound_up == pyrado.inf):
            raise pyrado.ValueErr(given=bound_up, eq_constraint="not +/- inf")

        self.bound_lo = bound_lo
        self.bound_up = bound_up

    def _convert_bounds(
        self, data: Union[to.Tensor, np.ndarray]
    ) -> Union[Tuple[to.Tensor, to.Tensor], Tuple[np.ndarray, np.ndarray]]:
        """
        Convert the bounds into the right type

        :param data: data that is later used for projecting
        :return: bounds casted to the type of data
        """
        if isinstance(data, to.Tensor) and isinstance(self.bound_lo, np.ndarray):
            bound_up, bound_lo = to.from_numpy(self.bound_up), to.from_numpy(self.bound_lo)
        elif isinstance(data, np.ndarray) and isinstance(self.bound_lo, to.Tensor):
            bound_up, bound_lo = self.bound_up.numpy(), self.bound_lo.numpy()
        else:
            bound_up, bound_lo = self.bound_up, self.bound_lo
        return bound_up, bound_lo

    def project_to(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        """
        Normalize every dimension individually using the stored explicit bounds and the L_1 norm.

        :param data: input to project to the unit space
        :return: element of the unit cube
        """
        if not isinstance(data, (to.Tensor, np.ndarray)):
            raise pyrado.TypeErr(given=data, expected_type=[to.Tensor, np.ndarray])

        # Convert if necessary
        bound_up, bound_lo = self._convert_bounds(data)

        span = bound_up - bound_lo
        span[span == 0] = 1.0  # avoid division by 0
        return (data - self.bound_lo) / span

    def project_back(self, data: Union[np.ndarray, to.Tensor]) -> Union[np.ndarray, to.Tensor]:
        """
        Revert the previous normalization using the stored explicit bounds

        :param data: input from the uni space
        :return: element of the original space
        """
        if not isinstance(data, (to.Tensor, np.ndarray)):
            raise pyrado.TypeErr(given=data, expected_type=[to.Tensor, np.ndarray])

        # Convert if necessary
        bound_up, bound_lo = self._convert_bounds(data)

        span = bound_up - bound_lo
        return span * data + bound_lo


def cov(x: to.Tensor, data_along_rows: bool = False):
    """
    Compute the covariance matrix given data.

    .. note::
        Only real valued matrices are supported

    :param x: matrix containing multiple observations of multiple variables
    :param data_along_rows: if `True` the variables are stacked along the columns, else they are along the rows
    :return: covariance matrix given the data
    """
    if x.dim() > 2:
        raise ValueError("m has more than 2 dimensions")
    if x.dim() < 2:
        x = x.view(1, -1)
    if data_along_rows and x.size(0) != 1:
        # Transpose if necessary
        x = x.t()

    num_samples = x.size(1)
    if num_samples < 2:
        raise pyrado.ShapeErr(msg="Need at least 2 samples to compute the covariance!")

    x -= to.mean(x, dim=1, keepdim=True)
    return x.matmul(x.t()).squeeze() / (num_samples - 1)


def explained_var(y_mdl: Union[np.ndarray, to.Tensor], y_obs: Union[np.ndarray, to.Tensor]) -> (np.ndarray, to.Tensor):
    """
    Calculate proportion of the variance "explained" by the model (see coefficient of determination R^2)
    .. note:: R^2 = 0.49 implies that 49% of the variability of the dependent variable has been accounted for.

    :param y_mdl: predictions from the model
    :param y_obs: observed data (ground truth)
    :return: proportion of the explained variance (i.e. R^2 value)
    """
    if isinstance(y_mdl, np.ndarray) and isinstance(y_obs, np.ndarray):
        ss_total = np.sum(np.power(y_obs - np.mean(y_obs), 2))  # proportional to the variance of the data
        ss_res = np.sum(np.power(y_obs - y_mdl, 2))  # proportional to the residual variance
        return 1.0 - ss_res / ss_total  # R^2

    elif isinstance(y_mdl, to.Tensor) and isinstance(y_obs, to.Tensor):
        ss_total = to.sum(to.pow(y_obs - to.mean(y_obs), 2))
        ss_res = to.sum(to.pow(y_obs - y_mdl, 2))
        return to.tensor(1.0) - ss_res / ss_total

    else:
        raise pyrado.TypeErr(given=y_mdl, expected_type=Union[np.ndarray, to.Tensor])


def logmeanexp(x: Union[to.Tensor, np.ndarray], dim: int = 0) -> Union[to.Tensor, np.ndarray]:
    r"""
    Numerically stable way to compute $\log \left( 1/N \sum_{i=1}^N \exp(x) \right)$

    :param x: input tensor
    :param dim: dimension to compute the logmeanexp along
    :return: $\log \left( 1/N \sum_{i=1}^N \exp(x) \right)$
    """
    if isinstance(x, to.Tensor):
        return to.logsumexp(x, dim=dim) - to.log(to.tensor(x.shape[dim], dtype=to.get_default_dtype()))
    elif isinstance(x, np.ndarray):
        return logsumexp(x, axis=dim) - np.log(np.array(x.shape[dim], dtype=np.float32))  # warning can be ignored
    else:
        raise pyrado.TypeErr(given=x, expected_type=[to.Tensor, np.ndarray])


def cosine_similarity(x: to.Tensor, y: to.Tensor) -> to.Tensor:
    r"""
    Compute the cosine similarity between two tensors $D_cos(x,y) = \frac{x^T y}{|x| \cdot |y|}$.

    :param x: input tensor
    :param y: input tensor
    :return: cosine similarity value
    """
    if not isinstance(x, to.Tensor):
        raise pyrado.TypeErr(given=x, expected_type=to.Tensor)
    if not isinstance(y, to.Tensor):
        raise pyrado.TypeErr(given=y, expected_type=to.Tensor)

    x_normed = normalize(x, order=2)
    y_normed = x_normed if y is x else normalize(y, order=2)
    return x_normed.dot(y_normed)


def rmse(
    x: Union[np.ndarray, to.Tensor], y: Union[np.ndarray, to.Tensor], dim: int = 0
) -> Union[np.ndarray, to.Tensor]:
    """
    Compute the root mean squared error between two inputs.

    :param x: input data, e.g. output of a neural network
    :param y: input data, e.g. target values for the output
    :param dim: dimension to compute the RMSE along
    :return:
    """
    if x.shape[dim] != y.shape[dim]:
        raise pyrado.ShapeErr(given=x, expected_match=y)

    if isinstance(x, to.Tensor) and isinstance(y, to.Tensor):
        return to.sqrt(to.mean(to.pow(x - y, 2), dim=dim))
    elif isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.sqrt(np.mean(np.power(x - y, 2), axis=dim))
    else:
        raise pyrado.TypeErr(msg="Both inputs need to be either a numpy array or a PyTorch tensor!")


def clamp(inp: to.Tensor, lo: to.Tensor, up: to.Tensor) -> to.Tensor:
    """
    Clip the entries of a tensor by the entries of a lower bound and an upper bound tensor.

    :param inp: input tensor
    :param lo: lower bound for each entry
    :param up: upper bound for each entry
    :return: clipped tensor
    """
    if not (lo < up).all():
        raise pyrado.ValueErr(msg="The lower bounds needs to be element-wise smaller than the upper bound.")
    return to.max(to.min(inp, up), lo)


def clamp_symm(inp: to.Tensor, up_lo: to.Tensor) -> to.Tensor:
    """
    Symmetrically clip the entries of a tensor by the entries of a tensor with only positive entries.
    One use case of this function is if you want to clip the parameter update by a ratio of the old parameters.

    :param inp: input tensor
    :param up_lo: upper and (negative of the) lower bound for each entry
    :return: clipped tensor
    """
    if not (up_lo > 0).all():
        raise pyrado.ValueErr(given=up_lo, g_constraint="0")
    return to.max(to.min(inp.clone(), up_lo), -up_lo)


def diff_coeffs(
    stencils: Union[list, np.ndarray], order: int, step_size: Union[int, float] = 1
) -> Tuple[np.ndarray, int]:
    r"""
    Compute the coefficients for discrete time numerical differentiation. These can later be convolved with the signal.

    .. seealso::
        https://en.wikipedia.org/wiki/Finite_difference_coefficient

    :param stencils: stencil points $s$, negative means in the past, e.g. [-1, 0, 1] for a central difference or
                     [-3, -1, 0] for a backward difference without equally space stencil points
    :param order: order of the derivative $d$, must be lower than the number of stencil points
    :param step_size: step size $h$
    :return: coefficients for computing the derivative of order `order` and the $O(h^{N-d})$
    """
    stencils = np.asarray(stencils).flatten()
    num_eqs = len(stencils)  # number of equations for the solver N
    stencils = np.unique(stencils)  # remove double stencils
    if len(stencils) != num_eqs:
        raise pyrado.ValueErr(msg="The array of stencil points must ponly contain unique elements!")
    if not stencils.ndim == 1:
        pyrado.ShapeErr(msg=f"The array of stencil points must be one-dimensional, but is {stencils}!")

    if not isinstance(order, int):
        raise pyrado.TypeErr(given=order, expected_type=int)
    if not 0 < order < num_eqs:
        raise pyrado.ValueErr(given=order, g_constraint="0", l_constraint=f"{num_eqs}")
    if not isinstance(step_size, (int, float)):
        raise pyrado.TypeErr(given=step_size, expected_type=[int, float])
    if step_size <= 0:
        raise pyrado.ValueErr(given=step_size, g_constraint="0")

    A = np.zeros((num_eqs, num_eqs))
    for idx in range(num_eqs):
        A[idx, :] = np.power(stencils, idx)
    b = np.zeros(num_eqs)
    b[order] = np.math.factorial(order) / step_size ** order

    coeffs = np.linalg.solve(A, b)
    return coeffs, num_eqs - order
