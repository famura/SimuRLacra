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
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import numpy as np
import torch as to
from init_args_serializer import Serializable

import pyrado
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environments.sim_base import SimEnv


class DomainParamTransform(EnvWrapper, ABC, Serializable):
    """
    Base class for all domain parameter transformations applied by the environment during setting and getting

    These transformations are useful for avoiding infeasible values such as negative masses. When set, a domain
    parameter is transformed using the `inverse()` function i.e. taking the exp() of the given values. The reasoning
    behind this is that some learning methods work on the set of real numbers, thus we make them learn in the
    transformed space, here the log-space, without telling them.
    """

    def __init__(self, wrapped_env: Union[SimEnv, EnvWrapper], mask: Union[List[str], Tuple[str]]):
        """
        Constructor

        :param wrapped_env: environment to wrap
        :param mask: every domain parameters which names are in this mask will be transformed. Capitalisation matters.
        """
        if not isinstance(wrapped_env, (SimEnv, EnvWrapper)):
            raise pyrado.TypeErr(given=wrapped_env, expected_type=(SimEnv, EnvWrapper))
        if not isinstance(mask, (list, tuple)):
            raise pyrado.TypeErr(given=wrapped_env, expected_type=(list, tuple))

        Serializable._init(self, locals())

        # Call EnvWrapper's constructor
        super().__init__(wrapped_env)

        if any(item not in wrapped_env.supported_domain_param for item in mask):
            raise pyrado.ValueErr(
                msg=f"The specified mask {mask} contains domain parameters that are not supported by the wrapped "
                f"environment! Here are the supported domain parameters {wrapped_env.supported_domain_param}."
            )
        self._mask = mask

    def forward(self, domain_param: dict) -> dict:
        """
        Map a domain parameter set to the transformed space.

        :param domain_param: domain parameter set in the original space
        :return: domain parameter set in the transformed space
        """
        for key, value in domain_param.items():
            domain_param[key] = self._forward(value) if key in self._mask else value
        return domain_param

    @staticmethod
    @abstractmethod
    def _forward(value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        """ Map a domain parameter value to the transformed space. """
        raise NotImplementedError

    def inverse(self, domain_param: dict) -> dict:
        """
        Map a domain parameter set back from the transformed space.

        :param domain_param: domain parameter set in the transformed space
        :return: domain parameter set in the original space
        """
        for key, value in domain_param.items():
            domain_param[key] = self._inverse(value) if key in self._mask else value
        return domain_param

    @staticmethod
    @abstractmethod
    def _inverse(value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        """ Map a domain parameter value back from the transformed space. """
        raise NotImplementedError

    @property
    def domain_param(self) -> dict:
        # The wrapped environment still has the un-transformed domain parameters.
        param = self._wrapped_env.domain_param
        self._set_wrapper_domain_param(param)  # see EnvWrapper
        return param

    @domain_param.setter
    def domain_param(self, domain_param: dict):
        # From the outside, to transformed domain parameter values are set, thus we transform them back before setting.
        self._get_wrapper_domain_param(domain_param)  # see EnvWrapper
        self._wrapped_env.domain_param = self.inverse(domain_param)


class LogDomainParamTransform(DomainParamTransform):
    """ Wrapper to make the domain parameters look like they are in log-space """

    @staticmethod
    def _forward(value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            return np.exp(value)
        elif isinstance(value, to.Tensor):
            return to.exp(value)
        else:
            return math.exp(value)

    @staticmethod
    def _inverse(value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            return np.log(value)
        elif isinstance(value, to.Tensor):
            return to.log(value)
        else:
            return math.log(value)


class SqrtDomainParamTransform(DomainParamTransform):
    """ Wrapper to make the domain parameters look like they are in sqrt-space """

    @staticmethod
    def _forward(value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            return np.power(value, 2)
        elif isinstance(value, to.Tensor):
            return to.pow(value, 2)
        else:
            return math.pow(value, 2)

    @staticmethod
    def _inverse(value: Union[int, float, np.ndarray, to.Tensor]) -> Union[int, float, np.ndarray, to.Tensor]:
        if isinstance(value, np.ndarray):
            return np.sqrt(value)
        elif isinstance(value, to.Tensor):
            return to.sqrt(value)
        else:
            return math.sqrt(value)
