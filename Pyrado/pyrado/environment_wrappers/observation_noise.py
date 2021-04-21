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

from typing import Optional, Union

import numpy as np
from init_args_serializer.serializable import Serializable

import pyrado
from pyrado.environment_wrappers.base import EnvWrapper, EnvWrapperObs
from pyrado.environments.sim_base import SimEnv


class GaussianObsNoiseWrapper(EnvWrapperObs, Serializable):
    """ Environment wrapper which adds normally distributed i.i.d. noise to all observations. """

    def __init__(
        self,
        wrapped_env: Union[SimEnv, EnvWrapper],
        noise_std: Union[list, np.ndarray],
        noise_mean: Optional[Union[list, np.ndarray]] = None,
    ):
        """
        :param wrapped_env: environment to wrap
        :param noise_std: list or numpy array for the standard deviation of the noise
        :param noise_mean: list or numpy array for the mean of the noise, by default all zeros, i.e. no bias
        """
        Serializable._init(self, locals())

        super().__init__(wrapped_env)

        # Parse noise specification
        self._std = np.array(noise_std)
        if not self._std.shape == self.obs_space.shape:
            raise pyrado.ShapeErr(given=self._std, expected_match=self.obs_space)
        if noise_mean is not None:
            self._mean = np.array(noise_mean)
            if not self._mean.shape == self.obs_space.shape:
                raise pyrado.ShapeErr(given=self._mean, expected_match=self.obs_space)
        else:
            self._mean = np.zeros(self.obs_space.shape)

    def _process_obs(self, obs: np.ndarray) -> np.ndarray:
        # Generate Gaussian noise sample
        noise = np.random.randn(*self.obs_space.shape) * self._std + self._mean

        # Add it to the observation
        return obs + noise

    def _save_domain_param(self, domain_param: dict):
        """
        Store the observation noise parameters in the domain parameter dict

        :param domain_param: domain parameter dict
        """
        domain_param["obs_noise_mean"] = self._mean
        domain_param["obs_noise_std"] = self._std

    def _load_domain_param(self, domain_param: dict):
        """
        Load the observation noise parameters from the domain parameter dict

        :param domain_param: domain parameter dict
        """
        if "obs_noise_mean" in domain_param:
            self._mean = np.array(domain_param["obs_noise_mean"])
            assert self._mean.shape == self.obs_space.shape
        if "obs_noise_std" in domain_param:
            self._std = np.array(domain_param["obs_noise_std"])
            assert self._std.shape == self.obs_space.shape
