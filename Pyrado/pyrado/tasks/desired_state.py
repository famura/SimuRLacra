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
from typing import Sequence, Callable

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.tasks.base import Task
from pyrado.tasks.utils import never_succeeded
from pyrado.tasks.reward_functions import RewFcn


class DesStateTask(Task):
    """ Task class for moving to a desired state. Operates on the error in state and action. """

    def __init__(self, env_spec: EnvSpec, state_des: np.ndarray, rew_fcn: RewFcn, success_fcn: Callable = None):
        """
        Constructor

        :param env_spec: environment specification of a simulated or real environment
        :param state_des: desired state a.k.a. goal state
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        :param success_fcn: function to determine if the task was solved, by default (`None`) this task runs endlessly
        """
        if not isinstance(env_spec, EnvSpec):
            raise pyrado.TypeErr(given=env_spec, expected_type=EnvSpec)
        if not isinstance(state_des, np.ndarray):
            raise pyrado.TypeErr(given=state_des, expected_type=np.ndarray)
        if not isinstance(rew_fcn, RewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=RewFcn)

        self._env_spec = env_spec
        self._state_des = state_des
        self._rew_fcn = rew_fcn
        self.success_fcn = success_fcn if success_fcn is not None else never_succeeded

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def state_des(self) -> np.ndarray:
        return self._state_des

    @state_des.setter
    def state_des(self, state_des: np.ndarray):
        if not isinstance(state_des, np.ndarray):
            raise pyrado.TypeErr(given=state_des, expected_type=np.ndarray)
        if not state_des.shape == self.state_des.shape:
            raise pyrado.ShapeErr(given=state_des, expected_match=self.state_des)
        self._state_des = state_des

    @property
    def rew_fcn(self) -> RewFcn:
        return self._rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        if not isinstance(rew_fcn, RewFcn):
            raise pyrado.TypeErr(given=rew_fcn, expected_type=RewFcn)
        self._rew_fcn = rew_fcn

    def reset(self, env_spec: EnvSpec, state_des: np.ndarray = None, **kwargs):
        """
        Reset the task.

        :param env_spec: environment specification
        :param state_des: new desired state a.k.a. goal state
        :param kwargs: keyword arguments forwarded to the reward function, e.g. the initial state
        """
        # Update the environment specification at every reset of the environment since the spaces could change
        self._env_spec = env_spec

        if state_des is not None:
            self._state_des = state_des

        # Some reward functions scale with the state and action bounds
        self._rew_fcn.reset(state_space=env_spec.state_space, act_space=env_spec.act_space, **kwargs)

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Operate on state and action errors
        err_state = self._state_des - state
        return self._rew_fcn(err_state, -act, remaining_steps)  # act_des = 0

    def has_succeeded(self, state: np.ndarray) -> bool:
        return self.success_fcn(self.state_des - state)


class RadiallySymmDesStateTask(DesStateTask):
    """
    Task class for moving to a desired state. Operates on the error in state and action.
    In contrast to DesStateTask, a subset of the state is radially symmetric, e.g. and angular position.
    """

    def __init__(
        self,
        env_spec: EnvSpec,
        state_des: np.ndarray,
        rew_fcn: RewFcn,
        idcs: Sequence[int],
        modulation: [float, np.ndarray] = 2 * np.pi,
        success_fcn: Callable = None,
    ):
        """
        Constructor

        :param env_spec: environment specification of a simulated or real environment
        :param state_des: desired state a.k.a. goal state
        :param rew_fcn: reward function, an instance of a subclass of RewFcn
        :param idcs: indices of the state dimension(s) to apply the modulation
        :param modulation: factor for the modulo operation, can be specified separately for every of `idcs`
        :param success_fcn: function to determine if the task was solved, by default (`None`) this task runs endlessly
        """
        super().__init__(env_spec, state_des, rew_fcn, success_fcn)

        self.idcs = idcs
        self.mod = modulation * np.ones(len(idcs))

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Modulate the state error
        err_state = self.state_des - state
        err_state[self.idcs] = np.fmod(err_state[self.idcs], self.mod)  # by default map to [-2pi, 2pi]

        # Look at the shortest path to the desired state i.e. desired angle
        err_state[err_state > np.pi] = 2 * np.pi - err_state[err_state > np.pi]  # e.g. 360 - (210) = 150
        err_state[err_state < -np.pi] = -2 * np.pi - err_state[err_state < -np.pi]  # e.g. -360 - (-210) = -150

        return self.rew_fcn(err_state, -act, remaining_steps)  # act_des = 0
