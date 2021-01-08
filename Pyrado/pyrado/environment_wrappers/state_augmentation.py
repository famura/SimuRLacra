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
from init_args_serializer import Serializable
from typing import Sequence

from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.spaces.box import BoxSpace


class StateAugmentationWrapper(EnvWrapper, Serializable):
    """ Augments the observation of the wrapped environment by its physics configuration """

    def __init__(self, wrapped_env: Env, domain_param: Sequence[str] = None, fixed: bool = False):
        """
        Constructor

        :param wrapped_env: the environment to be wrapped
        :param domain_param: list of domain parameter names to include in the observation, pass `None` to select all
        :param fixed: fix the parameters
        """
        Serializable._init(self, locals())

        EnvWrapper.__init__(self, wrapped_env)
        if domain_param is not None:
            self._params = domain_param
        else:
            self._params = list(inner_env(self.wrapped_env).domain_param.keys())
        self._nominal = inner_env(self.wrapped_env).get_nominal_domain_param()
        self._nominal = np.array([self._nominal[k] for k in self._params])
        self.fixed = fixed

    def _params_as_tensor(self):
        if self.fixed:
            return self._nominal
        else:
            return np.array([inner_env(self.wrapped_env).domain_param[k] for k in self._params])

    @property
    def obs_space(self):
        outer_space = self.wrapped_env.obs_space
        augmented_space = BoxSpace(0.5 * self._nominal, 1.5 * self._nominal, [self._nominal.shape[0]], self._params)
        return BoxSpace.cat((outer_space, augmented_space))

    def step(self, act: np.ndarray):
        obs, reward, done, info = self.wrapped_env.step(act)
        params = self._params_as_tensor()
        obs = np.concatenate((obs, params))
        return obs, reward, done, info

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        obs = self.wrapped_env.reset(init_state, domain_param)
        params = self._params_as_tensor()
        return np.concatenate((obs, params))

    @property
    def mask(self):
        return np.concatenate((np.zeros(self.wrapped_env.obs_space.flat_dim), np.ones(len(self._params))))

    @property
    def offset(self):
        return self.wrapped_env.obs_space.flat_dim

    def set_param(self, params):
        newp = dict()
        for key, value in zip(self._params, params):
            newp[key] = value.item()
        inner_env(self.wrapped_env).domain_param = newp

    def set_adv(self, params):
        for key, value in zip(self._params, params):
            inner_env(self.wrapped_env).domain_param[key] = self._nominal[key] + value

    @property
    def nominal(self):
        return self._nominal
