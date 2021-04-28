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

import pyrado
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import RewFcn, StateBasedRewFcn
from pyrado.utils.data_types import EnvSpec


class GoallessTask(Task):
    """Task which has no desired state or desired space, this runs endlessly"""

    def __init__(self, env_spec: EnvSpec, rew_fcn: RewFcn):
        """
        Constructor

        :param env_spec: environment specification
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        """
        self._env_spec = env_spec
        self._rew_fcn = rew_fcn

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def rew_fcn(self) -> RewFcn:
        return self._rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        if not isinstance(rew_fcn, RewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=RewFcn)
        self._rew_fcn = rew_fcn

    def reset(self, env_spec: EnvSpec, **kwargs):
        """
        Reset the task.

        :param env_spec: environment specification
        :param kwargs: keyword arguments forwarded to the reward function, e.g. the initial state
        """
        # Update the environment specification at every reset of the environment since the spaces could change
        self._env_spec = env_spec

        # Some reward functions scale with the state and action bounds
        self._rew_fcn.reset(state_space=env_spec.state_space, act_space=env_spec.act_space, **kwargs)

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Operate on state and actions
        return self.rew_fcn(state, act, remaining_steps)

    def has_succeeded(self, state: np.ndarray) -> bool:
        return False  # never succeed


class OptimProxyTask(Task):
    """Task for wrapping classical optimization problems a.k.a. (nonlinear) programming into Pyrado"""

    def __init__(self, env_spec: EnvSpec, rew_fcn: StateBasedRewFcn):
        """
        Constructor

        :param env_spec: environment specification
        :param rew_fcn: state-based reward function that maps the state to an scalar value
        """
        assert isinstance(rew_fcn, StateBasedRewFcn)

        self._env_spec = env_spec
        self._rew_fcn = rew_fcn

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def rew_fcn(self) -> StateBasedRewFcn:
        return self._rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: StateBasedRewFcn):
        if not isinstance(rew_fcn, StateBasedRewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=StateBasedRewFcn)
        self._rew_fcn = rew_fcn

    def reset(self, env_spec, **kwargs):
        # Nothing to do
        pass

    def step_rew(self, state: np.ndarray, act: np.ndarray = None, remaining_steps: int = None) -> float:
        # No dependency on the action or the time step here
        return self.rew_fcn(state)

    def has_succeeded(self, state: np.ndarray) -> bool:
        return False  # never succeed
