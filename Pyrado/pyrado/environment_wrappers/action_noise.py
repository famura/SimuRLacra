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
from typing import Union

import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environment_wrappers.base import EnvWrapperAct
from pyrado.environments.base import Env


class GaussianActNoiseWrapper(EnvWrapperAct, Serializable):
    """
    Environment wrapper which adds normally distributed i.i.d. noise to all action.
    This noise is independent for the potentially applied action-based exploration strategy.
    """

    def __init__(
        self, wrapped_env: Env, noise_mean: Union[float, np.ndarray] = None, noise_std: Union[float, np.ndarray] = None
    ):
        """
        Constructor

        :param wrapped_env: environment to wrap around (only makes sense for simulations)
        :param noise_mean: mean of the noise distribution
        :param noise_std: standard deviation of the noise distribution
        """
        Serializable._init(self, locals())

        # Invoke base constructor
        super().__init__(wrapped_env)

        # Parse noise specification
        if noise_mean is not None:
            self._mean = np.array(noise_mean)
            if not self._mean.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._mean, expected_match=self.act_space)
        else:
            self._mean = np.zeros(self.act_space.shape)
        if noise_std is not None:
            self._std = np.array(noise_std)
            if not self._std.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._noise_std, expected_match=self.act_space)
        else:
            self._std = np.zeros(self.act_space.shape)

    def _process_act(self, act: np.ndarray) -> np.ndarray:
        # Generate gaussian noise values
        noise = np.random.randn(*self.act_space.shape) * self._std + self._mean  # * to unsqueeze the tuple

        # Add it to the action
        return act + noise

    def _set_wrapper_domain_param(self, domain_param: dict):
        """
        Store the action noise parameters in the domain parameter dict

        :param domain_param: domain parameter dict
        """
        domain_param["act_noise_mean"] = self._mean
        domain_param["act_noise_std"] = self._std

    def _get_wrapper_domain_param(self, domain_param: dict):
        """
        Load the action noise parameters from the domain parameter dict

        :param domain_param: domain parameter dict
        """
        if "act_noise_mean" in domain_param:
            self._noise_mean = np.array(domain_param["act_noise_mean"])
            if not self._noise_mean.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._noise_mean, expected_match=self.act_space)
        if "act_noise_std" in domain_param:
            self._noise_std = np.array(domain_param["act_noise_std"])
            if not self._noise_std.shape == self.act_space.shape:
                raise pyrado.ShapeErr(given=self._noise_std, expected_match=self.act_space)
