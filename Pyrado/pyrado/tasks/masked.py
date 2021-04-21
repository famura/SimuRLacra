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

from pyrado.spaces.empty import EmptySpace
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import RewFcn
from pyrado.utils.data_types import EnvSpec


class MaskedTask(Task):
    """ Task using only a subset of state and actions """

    def __init__(
        self,
        env_spec: EnvSpec,
        wrapped_task: Task,
        state_idcs: Union[str, int],
        action_idcs: Optional[Union[str, int]] = None,
    ):
        """
        Constructor

        :param env_spec: environment specification
        :param wrapped_task: task for the selected part of the state-action space
        :param state_idcs: indices of the selected states
        :param action_idcs: indices of the selected actions
        """
        self._env_spec = env_spec
        self._wrapped_task = wrapped_task
        self._state_idcs = state_idcs
        self._action_idcs = action_idcs

        # Written by reset
        self._state_mask = None
        self._action_mask = None
        self.reset(env_spec)

    @property
    def env_spec(self) -> EnvSpec:
        return self._env_spec

    @property
    def wrapped_task(self) -> Task:
        return self._wrapped_task

    @property
    def state_des(self) -> np.ndarray:
        # The desired state is NaN for masked entries.
        full = np.full(self.env_spec.state_space.shape, np.nan)
        full[self._state_mask] = self._wrapped_task.state_des
        return full

    @state_des.setter
    def state_des(self, state_des: np.ndarray):
        self._wrapped_task.state_des = state_des[self._state_mask]

    @property
    def rew_fcn(self) -> RewFcn:
        return self._wrapped_task.rew_fcn

    @rew_fcn.setter
    def rew_fcn(self, rew_fcn: RewFcn):
        self._wrapped_task.rew_fcn = rew_fcn

    def reset(self, env_spec: EnvSpec, **kwargs):
        self._env_spec = env_spec

        # Determine the masks
        if self._state_idcs is not None:
            self._state_mask = env_spec.state_space.create_mask(self._state_idcs)
        else:
            self._state_mask = np.ones(env_spec.state_space.shape, dtype=np.bool_)
        if self._action_idcs is not None:
            self._action_mask = env_spec.act_space.create_mask(self._action_idcs)
        else:
            self._action_mask = np.ones(env_spec.act_space.shape, dtype=np.bool_)

        # Pass masked state and masked action
        self._wrapped_task.reset(
            env_spec=EnvSpec(
                env_spec.obs_space,
                env_spec.act_space.subspace(self._action_mask),
                env_spec.state_space.subspace(self._state_mask)
                if env_spec.state_space is not EmptySpace
                else EmptySpace,
            ),
            **kwargs,
        )

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        # Pass masked state and masked action
        return self._wrapped_task.step_rew(state[self._state_mask], act[self._action_mask], remaining_steps)

    def final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        # Pass masked state and masked action
        return self._wrapped_task.final_rew(state[self._state_mask], remaining_steps)

    def has_succeeded(self, state: np.ndarray) -> bool:
        # Pass masked state and masked action
        return self._wrapped_task.has_succeeded(state[self._state_mask])

    def has_failed(self, state: np.ndarray) -> bool:
        # Pass masked state and masked action
        return self._wrapped_task.has_failed(state[self._state_mask])

    def is_done(self, state: np.ndarray) -> bool:
        # Pass masked state and masked action
        return self._wrapped_task.is_done(state[self._state_mask])
