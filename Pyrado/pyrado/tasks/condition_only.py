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

from typing import Callable

import numpy as np

import pyrado
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import RewFcn, ZeroPerStepRewFcn
from pyrado.utils.data_types import EnvSpec


class ConditionOnlyTask(Task):
    """
    Task class which yields zero reward at every time and terminates if the given function is true.
    This class is intended to be wrapped by `FinalRewTask`.
    """

    def __init__(self, env_spec: EnvSpec, condition_fcn: Callable, is_success_condition: bool):
        """
        Constructor

        :usage:
        .. code-block:: python

            task = FinalRewTask(
                       ConditionOnlyTask(<some EnvSpec>, <some Callable>, <True or False>),
                       mode=FinalRewMode(time_dependent=True)
            )

        :param env_spec: environment specification of a simulated or real environment
        :param condition_fcn: function to determine if the task was solved, by default (`None`) this task runs endlessly
        :param is_success_condition: if `True` the `condition_fcn` returns `True` for a success,
                                     if `False` the `condition_fcn` returns `True` for a failure
        """
        if not isinstance(env_spec, EnvSpec):
            raise pyrado.TypeErr(given=env_spec, expected_type=EnvSpec)
        if not callable(condition_fcn):
            raise pyrado.TypeErr(given=condition_fcn, expected_type=Callable)

        self._env_spec = env_spec
        self.condition_fcn = condition_fcn
        self.is_success_condition = is_success_condition

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def rew_fcn(self) -> RewFcn:
        # To expose that this task yields zero reward per step
        return ZeroPerStepRewFcn()

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        # Should not be called
        raise NotImplementedError

    def reset(self, env_spec: EnvSpec, condition_fcn: Callable = None, is_success_condition: bool = None, **kwargs):
        """
        Reset the task.

        :param env_spec: environment specification
        :param condition_fcn: function to determine if the task was solved, by default (`None`) this task runs endlessly
        :param is_success_condition: if `True` the `condition_fcn` returns `True` for a success,
                                     if `False` the `condition_fcn` returns `True` for a failure
        """
        # Update the environment specification at every reset of the environment since the spaces could change
        self._env_spec = env_spec

        if condition_fcn is not None:
            self.condition_fcn = condition_fcn
        if is_success_condition is not None:
            self.is_success_condition = is_success_condition

    def step_rew(self, state: np.ndarray = None, act: np.ndarray = None, remaining_steps: int = None) -> float:
        return 0.0

    def has_succeeded(self, state: np.ndarray) -> bool:
        if self.is_success_condition:
            # Use given condition function to determine success
            return self.condition_fcn(state)
        else:
            # Ignore condition function
            return False

    def has_failed(self, state: np.ndarray) -> bool:
        if self.is_success_condition:
            # Ignore condition function
            return super().has_failed(state)
        else:
            # Use given condition function to determine failure
            return self.condition_fcn(state)
