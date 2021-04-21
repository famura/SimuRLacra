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
from typing import Union
from warnings import warn

import torch as to
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal

import pyrado


class DiagNormalNoise(nn.Module):
    """ Module for learnable additive Gaussian noise with a diagonal covariance matrix """

    def __init__(
        self,
        noise_dim: [int, tuple],
        std_init: [float, int, to.Tensor],
        std_min: [float, to.Tensor] = 1e-3,
        train_mean: bool = False,
        learnable: bool = True,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param noise_dim: number of dimension
        :param std_init: initial standard deviation for the exploration noise
        :param std_min: minimal standard deviation for the exploration noise
        :param train_mean: `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        :param use_cuda: `True` to move the module to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(std_init, (float, int, to.Tensor)):
            raise pyrado.TypeErr(given=std_init, expected_type=[float, int, to.Tensor])
        if isinstance(std_init, to.Tensor) and not std_init.size() == noise_dim:
            raise pyrado.ShapeErr(given=std_init, expected_match=to.empty(noise_dim))
        if not (
            isinstance(std_init, (float, int)) and std_init > 0 or isinstance(std_init, to.Tensor) and all(std_init > 0)
        ):
            raise pyrado.ValueErr(given=std_init, g_constraint="0")
        if not isinstance(std_min, (float, to.Tensor)):
            raise pyrado.TypeErr(given=std_min, expected_type=[float, to.Tensor])

        # Call torch.nn.Module's constructor
        super().__init__()

        if not use_cuda:
            self._device = "cpu"
        elif use_cuda and to.cuda.is_available():
            self._device = "cuda"
        elif use_cuda and not to.cuda.is_available():
            warn("Tried to run on CUDA, but it is not available. Falling back to CPU.")
            self._device = "cpu"

        # Register parameters
        if learnable:
            self.log_std = nn.Parameter(to.Tensor(noise_dim), requires_grad=True)
            self.mean = nn.Parameter(to.Tensor(noise_dim), requires_grad=True) if train_mean else None
        else:
            self.log_std = to.empty(noise_dim)
            self.mean = None

        # Initialize parameters
        self.log_std_init = to.log(to.tensor(std_init)) if isinstance(std_init, float) else to.log(std_init)
        self.std_min = (
            to.tensor(std_min, dtype=to.get_default_dtype()) if isinstance(std_min, (float, int)) else std_min
        )
        self.log_std_init = self.log_std_init.to(device=self.device)
        self.std_min = self.std_min.to(device=self.device)
        if not isinstance(self.log_std_init, to.Tensor):
            raise pyrado.TypeErr(given=self.log_std_init, expected_type=to.Tensor)
        if not isinstance(self.std_min, to.Tensor):
            raise pyrado.TypeErr(given=self.std_min, expected_type=to.Tensor)

        self.reset_expl_params()
        self.to(self.device)

    @property
    def device(self) -> str:
        """ Get the device (CPU or GPU) on which the policy is stored. """
        return self._device

    @property
    def std(self) -> to.Tensor:
        """ Get the untransformed standard deviation from the log-transformed. """
        return to.exp(self.log_std)

    @std.setter
    def std(self, std: to.Tensor):
        """
        Set the log-transformed standard deviation given the untransformed.
        This is useful if the `std` should be a parameter for the optimizer, since the optimizer could set invalid
        values for the standard deviation.
        """
        self.log_std.data = to.log(to.max(std, self.std_min))

    def reset_expl_params(self):
        """ Reset all parameters of the exploration strategy. """
        if self.mean is not None:
            self.mean.data.zero_()
        self.log_std.data.copy_(self.log_std_init)

    def adapt(self, mean: to.Tensor = None, std: Union[to.Tensor, float] = None):
        """
        Adapt the mean and the variance of the noise on the action or parameters.
        Use `None` to leave one of the parameters at their current value.

        :param mean: exploration strategy's new mean
        :param std: exploration strategy's new standard deviation
        """
        if not (isinstance(mean, to.Tensor) or mean is None):
            raise pyrado.TypeErr(given=mean, expected_type=to.Tensor)
        if not (isinstance(std, to.Tensor) and (std >= 0).all() or std is None):
            raise pyrado.TypeErr(msg=f"The std must be a Tensor with all elements > 0 or None, but it is {std}!")

        if mean is not None:
            assert self.mean is not None, "Can not change fixed zero mean!"
            if not mean.shape == self.mean.shape:
                raise pyrado.ShapeErr(given=mean, expected_match=self.mean)
            self.mean.data = mean
        if std is not None:
            if not std.shape == self.log_std.shape:
                raise pyrado.ShapeErr(given=std, expected_match=self.std)
            self.std = std

    def forward(self, value: to.Tensor) -> Normal:
        """
        Return the noise distribution for a specific noise-free value.

        :param value: value to evaluate the distribution around
        :return: noise distribution
        """
        mean = value if self.mean is None else value + self.mean
        return Normal(mean, self.std)

    def get_entropy(self) -> to.Tensor:
        """
        Get the exploration distribution's entropy.
        The entropy of a normal distribution is independent of the mean.

        :return: entropy value
        """
        return 0.5 + 0.5 * to.log(to.tensor(2 * math.pi)) + 0.5 * to.log(to.prod(to.pow(self.std, 2)))


class FullNormalNoise(nn.Module):
    """ Module for learnable additive Gaussian noise with a full covariance matrix """

    def __init__(
        self,
        noise_dim: [int, tuple],
        std_init: [float, int, to.Tensor],
        std_min: [float, to.Tensor] = 1e-3,
        train_mean: bool = False,
        learnable: bool = True,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param noise_dim: number of dimension
        :param std_init: initial standard deviation for the exploration noise
        :param std_min: minimal standard deviation for the exploration noise
        :param train_mean: `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        :param use_cuda: `True` to move the module to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(std_init, (float, int, to.Tensor)):
            raise pyrado.TypeErr(given=std_init, expected_type=[float, int, to.Tensor])
        if isinstance(std_init, to.Tensor) and not std_init.size() == noise_dim:
            raise pyrado.ShapeErr(given=std_init, expected_match=to.empty(noise_dim))
        if not (isinstance(std_init, float) and std_init > 0 or isinstance(std_init, to.Tensor) and all(std_init > 0)):
            raise pyrado.ValueErr(given=std_init, g_constraint="0")
        if not isinstance(std_min, (float, to.Tensor)):
            raise pyrado.TypeErr(given=std_min, expected_type=[float, to.Tensor])
        if not (isinstance(std_min, float) and std_min > 0 or isinstance(std_min, to.Tensor) and all(std_min > 0)):
            raise pyrado.ValueErr(given=std_min, g_constraint="0")

        # Call torch.nn.Module's constructor
        super().__init__()

        if not use_cuda:
            self._device = "cpu"
        elif use_cuda and to.cuda.is_available():
            self._device = "cuda"
        elif use_cuda and not to.cuda.is_available():
            warn("Tried to run on CUDA, but it is not available. Falling back to CPU.")
            self._device = "cpu"

        # Register parameters
        if learnable:
            self.cov = nn.Parameter(to.Tensor(noise_dim, noise_dim), requires_grad=True)
            self.mean = nn.Parameter(to.Tensor(noise_dim), requires_grad=True) if train_mean else None
        else:
            self.cov = to.empty(noise_dim, noise_dim)
            self.mean = None

        # Initialize parameters
        self.cov_init = (
            std_init ** 2 * to.eye(noise_dim) if isinstance(std_init, float) else to.diag(to.pow(std_init, 2))
        )
        self.std_min = to.tensor(std_min) if isinstance(std_min, float) else std_min
        self.std_min = self.std_min.to(device=self.device)
        if not isinstance(self.cov_init, to.Tensor):
            raise pyrado.TypeErr(given=self.cov_init, expected_type=to.Tensor)
        if not isinstance(self.std_min, to.Tensor):
            raise pyrado.TypeErr(given=self.std_min, expected_type=to.Tensor)

        self.reset_expl_params()
        self.to(self.device)

    @property
    def device(self) -> str:
        """ Get the device (CPU or GPU) on which the policy is stored. """
        return self._device

    @property
    @to.no_grad()
    def std(self) -> to.Tensor:
        """ Get the standard deviations from the internal covariance matrix. """
        return to.sqrt(self.cov.diag())

    def reset_expl_params(self):
        """ Reset all parameters of the exploration strategy. """
        if self.mean is not None:
            self.mean.data.zero_()
        self.cov.data.copy_(self.cov_init)

    def adapt(self, mean: to.Tensor = None, cov: Union[to.Tensor, float] = None):
        """
        Adapt the mean and the variance of the noise on the action or parameters.
        Use `None` to leave one of the parameters at their current value.

        :param mean: exploration strategy's new mean
        :param cov: exploration strategy's new standard deviation
        """
        if not (isinstance(mean, to.Tensor) or mean is None):
            raise pyrado.TypeErr(given=mean, expected_type=to.Tensor)
        if not (isinstance(cov, to.Tensor) or cov is None):
            raise pyrado.TypeErr(given=cov, expected_type=to.Tensor)

        if mean is not None:
            assert self.mean is not None, "Can not change fixed zero mean!"
            if not mean.shape == self.mean.shape:
                raise pyrado.ShapeErr(given=mean, expected_match=self.mean)
            self.mean.data = mean

        if cov is not None:
            if not cov.shape == self.cov.shape:
                raise pyrado.ShapeErr(given=cov, expected_match=self.cov)
            self.cov.data = cov

            # Check the eigenvalues of the new matrix and make if positive definite if necessary
            eigs = to.symeig(self.cov.data, eigenvectors=False)[0]
            if not min(eigs > 0.0):
                repair_diag = min(eigs - 1e-3) * to.eye(self.cov.shape[0])
                self.cov.data -= repair_diag  # shifting the eigenvalues
                warn("The covariance matrix has not been positive definite.", UserWarning)

            # Check if the diagonal elements of the new covariance matrix are neither too small nor too big
            for i in range(cov.shape[0]):
                self.cov.data[i, i] = to.max(self.cov.data[i, i], self.std_min ** 2)

    def forward(self, value: to.Tensor) -> MultivariateNormal:
        """
        Return the noise distribution for a specific noise-free value.

        :param value: value to evaluate the distribution around
        :return: noise distribution
        """
        mean = value if self.mean is None else value + self.mean
        return MultivariateNormal(mean, self.cov)

    def get_entropy(self) -> to.Tensor:
        """
        Get the exploration distribution's entropy.
        The entropy of a normal distribution is independent of the mean.

        :return: entropy value
        """
        # det_cov = to.cholesky(self.cov).diag().prod()  # cholesky returns lower triangular matrix
        det_cov = to.prod(to.symeig(self.cov.data, eigenvectors=False)[0])
        return self.cov.shape[0] / 2.0 * (1.0 + to.log(to.tensor(2 * math.pi))) + to.log(det_cov) / 2.0
