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

import torch as to
import torch.nn as nn
from torch.distributions import Uniform
from warnings import warn

import pyrado


class UniformNoise(nn.Module):
    """ Module for learnable additive uniform noise """

    def __init__(self,
                 use_cuda: bool,
                 noise_dim: [int, tuple],
                 halfspan_init: [float, to.Tensor],
                 halfspan_min: [float, to.Tensor] = 0.01,
                 train_mean: bool = False,
                 learnable: bool = True):
        """
        Constructor

        :param use_cuda: `True` to move the module to the GPU, `False` (default) to use the CPU
        :param noise_dim: number of dimension
        :param halfspan_init: initial value of the half interval for the exploration noise
        :param halfspan_min: minimal value of the half interval for the exploration noise
        :param train_mean: `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param learnable: `True` if the parameters should be tuneable (default), `False` for shallow use (just sampling)
        """
        if not isinstance(halfspan_init, (float, to.Tensor)):
            raise pyrado.TypeErr(given=halfspan_init, expected_type=[float, to.Tensor])
        if not (isinstance(halfspan_init, float) and halfspan_init > 0 or
                isinstance(halfspan_init, to.Tensor) and all(halfspan_init > 0)):
            raise pyrado.ValueErr(given=halfspan_init, g_constraint='0')
        if not isinstance(halfspan_min, (float, to.Tensor)):
            raise pyrado.TypeErr(given=halfspan_min, expected_type=[float, to.Tensor])
        if not (isinstance(halfspan_min, float) and halfspan_min > 0 or
                isinstance(halfspan_min, to.Tensor) and all(halfspan_min > 0)):
            raise pyrado.ValueErr(given=halfspan_min, g_constraint='0')

        # Call torch.nn.Module's constructor
        super().__init__()

        if not use_cuda:
            self._device = 'cpu'
        elif use_cuda and to.cuda.is_available():
            self._device = 'cuda'
        elif use_cuda and not to.cuda.is_available():
            warn('Tried to run on CUDA, but it is not available. Falling back to CPU.')
            self._device = 'cpu'

        # Register parameters
        if learnable:
            self.log_halfspan = nn.Parameter(to.Tensor(noise_dim), requires_grad=True)
            self.mean = nn.Parameter(to.Tensor(noise_dim), requires_grad=True) if train_mean else None
        else:
            self.log_halfspan = to.empty(noise_dim)
            self.mean = None

        # Initialize parameters
        self.log_halfspan_init = to.log(to.tensor(halfspan_init)) if isinstance(halfspan_init, float) else to.log(
            halfspan_init)
        self.halfspan_min = to.tensor(halfspan_min) if isinstance(halfspan_min, float) else halfspan_min
        if not isinstance(self.log_halfspan_init, to.Tensor):
            raise pyrado.TypeErr(given=self.log_halfspan_init, expected_type=to.Tensor)
        if not isinstance(self.halfspan_min, to.Tensor):
            raise pyrado.TypeErr(given=self.halfspan_min, expected_type=to.Tensor)

        self.reset_expl_params()
        self.to(self.device)

    @property
    def device(self) -> str:
        """ Get the device (CPU or GPU) on which the policy is stored. """
        return self._device

    @property
    def halfspan(self) -> to.Tensor:
        """ Get the untransformed standard deviation vector given the log-transformed. """
        return to.max(to.exp(self.log_halfspan), self.halfspan_min)

    @halfspan.setter
    def halfspan(self, halfspan: to.Tensor):
        """
        Set the log-transformed half interval vector given the untransformed.
        This is useful if the `halfspan` should be a parameter for the optimizer, since the optimizer could set invalid
        values for the half interval span.
        """
        self.log_halfspan.data = to.log(to.max(halfspan, self.halfspan_min))

    def reset_expl_params(self):
        """ Reset all parameters of the exploration strategy. """
        if self.mean is not None:
            self.mean.data.zero_()
        self.log_halfspan.data.copy_(self.log_halfspan_init)

    def adapt(self, mean: to.Tensor = None, halfspan: [to.Tensor, float] = None):
        """
        Adapt the mean and the half interval span of the noise on the action or parameters.
        Use `None` to leave one of the parameters at their current value.

        :param mean: exploration strategy's new mean
        :param halfspan: exploration strategy's new half interval span
        """
        if not (isinstance(mean, to.Tensor) or mean is None):
            raise pyrado.TypeErr(given=mean, expected_type=to.Tensor)
        if not (isinstance(halfspan, to.Tensor) and (halfspan >= 0).all() or halfspan is None):
            raise pyrado.TypeErr(msg='The halfspan must be a Tensor with all elements > 0 or None!')
        if mean is not None:
            assert self.mean is not None, 'Can not change fixed zero mean!'
            if not mean.shape == self.mean.shape:
                raise pyrado.ShapeErr(given=mean, expected_match=self.mean)
            self.mean.data = mean
        if halfspan is not None:
            if not halfspan.shape == self.log_halfspan.shape:
                raise pyrado.ShapeErr(given=halfspan, expected_match=self.halfspan)
            self.halfspan = halfspan

    def forward(self, value: to.Tensor) -> Uniform:
        """
        Return the noise distribution for a specific noise-free value.

        :param value: value to evaluate the distribution around
        :return: noise distribution
        """
        mean = value if self.mean is None else value + self.mean
        return Uniform(low=mean - self.halfspan, high=mean + self.halfspan)

    def get_entropy(self) -> to.Tensor:
        """
        Get the exploration distribution's entropy.
        The entropy of a uniform distribution is independent of the mean.

        :return: entropy value
        """
        return to.log(2*self.halfspan)
