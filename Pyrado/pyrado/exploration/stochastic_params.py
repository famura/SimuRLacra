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
from typing import Sequence
from warnings import warn

import torch as to

import pyrado
from pyrado.exploration.normal_noise import DiagNormalNoise, FullNormalNoise
from pyrado.sampling.hyper_sphere import sample_from_hyper_sphere_surface


class StochasticParamExplStrat(ABC):
    """Exploration strategy which samples policy parameters from a distribution"""

    def __init__(self, param_dim: int):
        """
        Constructor

        :param param_dim: number of policy parameters
        """
        self.param_dim = param_dim

    @abstractmethod
    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        """
        Sample one set of policy parameters from the current distribution.

        :param nominal_params: parameter set (1-dim tensor) to sample around
        :return: sampled parameter set (1-dim tensor)
        """
        raise NotImplementedError

    def sample_param_sets(
        self, nominal_params: to.Tensor, num_samples: int, include_nominal_params: bool = False
    ) -> to.Tensor:
        """
        Sample multiple sets of policy parameters from the current distribution.

        :param nominal_params: parameter set (1-dim tensor) to sample around
        :param num_samples: number of parameter sets
        :param include_nominal_params: `True` to include the nominal parameter values as first parameter set
        :return: policy parameter sets as NxP or (N+1)xP tensor where N is the number samples and P is the number of
                 policy parameters
        """
        ps = [self.sample_param_set(nominal_params) for _ in range(num_samples)]
        if include_nominal_params:
            ps.insert(0, nominal_params)
        return to.stack(ps)


class SymmParamExplStrat(StochasticParamExplStrat):
    """
    Wrap a parameter exploration strategy to enforce symmetric sampling.
    The function `sample_param_sets` will always return an even number of parameters, and it's guaranteed that
    ps[:len(ps)//2] == -ps[len(ps)//2:]
    """

    def __init__(self, wrapped: StochasticParamExplStrat):
        """
        Constructor

        :param wrapped: exploration strategy to wrap around
        """
        super().__init__(wrapped.param_dim)
        self.wrapped = wrapped

    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        # Should not be done, but fail gracefully
        warn("Called sample_param_set on SymmParamExplStrat, which will still return only one param set", stacklevel=2)
        return self.wrapped.sample_param_set(nominal_params)

    def sample_param_sets(
        self, nominal_params: to.Tensor, num_samples: int, include_nominal_params: bool = False
    ) -> to.Tensor:
        # Adjust sample size to be even
        if num_samples % 2 != 0:
            num_samples += 1

        # Sample one half
        pos_half = self.wrapped.sample_param_sets(nominal_params, num_samples // 2)

        # Mirror around nominal params for the other half
        # hp = nom + eps => eps = hp - nom
        # hn = nom - eps = nom - (hp - nom) = 2*nom - hp
        neg_half = 2.0 * nominal_params - pos_half
        parts = [pos_half, neg_half]

        # Add nominal params if requested
        if include_nominal_params:
            parts.insert(0, nominal_params.view(1, -1))
        return to.cat(parts)

    def __getattr__(self, name: str):
        """
        Forward unknown attributes to wrapped strategy.

        :param name: name of the attribute
        """
        if name == "wrapped":
            # Can happen while unpickling, when the wrapped attr isn't set yet. Since it should normally reside in
            # __dict__, it's correct to raise a missing attr here
            raise AttributeError
        # The getattr method raises an AttributeError if not found, just as this method should
        return getattr(self.wrapped, name)


class NormalParamNoise(StochasticParamExplStrat):
    """Sampling parameters from a normal distribution"""

    def __init__(
        self,
        param_dim: int,
        full_cov: bool = False,
        std_init: float = 1.0,
        std_min: [float, Sequence[float]] = 0.01,
        train_mean: bool = False,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param param_dim: number of policy parameters
        :param full_cov: use a full covariance matrix or a diagonal covariance matrix (independent random variables)
        :param std_init: initial standard deviation for the noise distribution
        :param std_min: minimal standard deviation for the exploration noise
        :param train_mean: set `True` if the noise should have an adaptive nonzero mean, `False` otherwise
        :param use_cuda: `True` to move the module to the GPU, `False` (default) to use the CPU
        """
        # Call the StochasticParamExplStrat's constructor
        super().__init__(param_dim)

        if full_cov:
            self._noise = FullNormalNoise(
                noise_dim=param_dim, std_init=std_init, std_min=std_min, train_mean=train_mean, use_cuda=use_cuda
            )
        else:
            self._noise = DiagNormalNoise(
                noise_dim=param_dim, std_init=std_init, std_min=std_min, train_mean=train_mean, use_cuda=use_cuda
            )

    def reset_expl_params(self, *args, **kwargs):
        self._noise.reset_expl_params(*args, **kwargs)

    def adapt(self, *args, **kwargs):
        self._noise.adapt(*args, **kwargs)

    def get_entropy(self, *args, **kwargs):
        return self._noise.get_entropy(*args, **kwargs)

    @property
    def noise(self) -> [FullNormalNoise, DiagNormalNoise]:
        """Get the exploration noise."""
        return self._noise

    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        return self._noise(nominal_params).sample()

    def sample_param_sets(
        self, nominal_params: to.Tensor, num_samples: int, include_nominal_params: bool = False
    ) -> to.Tensor:
        # Sample all exploring policy parameter at once
        ps = self._noise(nominal_params).sample((num_samples,))
        if include_nominal_params:
            ps = to.cat([nominal_params.view(1, -1), ps])
        return ps

    @property
    def std(self):
        return self._noise.std

    @std.setter
    def std(self, value):
        self._noise.std = value

    @property
    def cov(self):
        return self._noise.cov

    @cov.setter
    def cov(self, value):
        self._noise.cov = value


class HyperSphereParamNoise(StochasticParamExplStrat):
    """Sampling parameters from a normal distribution"""

    def __init__(self, param_dim: int, expl_r_init: float = 1.0):
        """
        Constructor

        :param param_dim: number of policy parameters
        :param expl_r_init: initial radius of the hyper-sphere
        """
        # Call the base class constructor
        super().__init__(param_dim)

        self._r_init = expl_r_init
        self._r = expl_r_init

    @property
    def r(self) -> float:
        """Get the radius of the hypersphere."""
        return self._r

    def reset_expl_params(self):
        """Reset all parameters of the exploration strategy."""
        self._r = self._r_init

    def adapt(self, r: float):
        """Set a new radius for the hyper sphere from which the policy parameters are sampled."""
        if not r > 0.0:
            pyrado.ValueErr(given=r, g_constraint="0")
        self._r = r

    def sample_param_set(self, nominal_params: to.Tensor) -> to.Tensor:
        return nominal_params + self._r * sample_from_hyper_sphere_surface(self.param_dim, "normal")
