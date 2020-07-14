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
import numpy as np
from typing import Any

import pyrado
from pyrado.spaces.base import Space
from pyrado.utils import get_class_name
from pyrado.utils.data_types import EnvSpec
from pyrado.tasks.reward_functions import RewFcn


class Task(ABC):
    """
    Base class for all tasks in Pyrado.
    A task contains a desired state, a reward function, and a step function.
    The task also checks if the environment is done. Every environment should have exactly one task at a time.
    """

    @property
    def env_spec(self) -> EnvSpec:
        """ Get the specification of environment the task is in. """
        raise NotImplementedError

    @property
    def state_des(self) -> np.ndarray:
        """
        Get the desired state (same dimensions as the environment's state).
        Only override this if the task has a desired state.
        """
        raise AttributeError(f'{get_class_name(self)} has no desired state.')

    @state_des.setter
    def state_des(self, state_des: np.ndarray):
        """
        Set the desired state (same dimensions as the environment's state).
        Only override this if the task has a desired state.
        """
        raise AttributeError(f'{get_class_name(self)} has no desired state.')

    @property
    def space_des(self) -> Space:
        """
        Get the desired state (same dimensions as the environment's state).
        Only override this if the task has a desired state.
        """
        raise AttributeError(f'{get_class_name(self)} has no desired space.')

    @space_des.setter
    def space_des(self, space_des: np.ndarray):
        """
        Set the desired state (same dimensions as the environment's state).
        Only override this if the task has a desired state.
        """
        raise AttributeError(f'{get_class_name(self)} has no desired space.')

    @property
    @abstractmethod
    def rew_fcn(self) -> RewFcn:
        """ Get the reward function. """
        raise NotImplementedError

    @rew_fcn.setter
    @abstractmethod
    def rew_fcn(self, rew_fcn: RewFcn):
        """ Set the reward function. """
        raise NotImplementedError

    def reset(self, **kwargs: Any):
        """
        Reset the task.
        Since the environment specification may change at every reset of the environment, we have to reset the task.
        This might also include resetting the members of the reward function if there are any.

        :param kwargs: optional arguments e.g. environment specification or new desired state
        """
        raise NotImplementedError

    @abstractmethod
    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        """
        Get the step reward, e.g. from a function of the states and actions.

        .. note::
            It is strongly recommended to call this method every environment step.

        :param state: current state
        :param act: current action
        :param remaining_steps: number of time steps left in the episode
        :return rew: current reward
        """
        raise NotImplementedError

    def final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Get the final reward, e.g. bonus for success or a malus for failure.
        This function loops through all tasks (unfolding the wrappers) and calls their `compute_final_rew` method.

        :param state: current state forwarded to `compute_final_rew`
        :param remaining_steps: number of time steps left in the episode forwarded to `compute_final_rew`
        :return rew: summed final reward
        """
        sum_final_rew = 0.
        for t in all_tasks(self):
            sum_final_rew += t.compute_final_rew(state, remaining_steps)
        return sum_final_rew

    def compute_final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the final reward, e.g. bonus for success or a malus for failure, for a single task.

        .. note::
            This function should only be overwritten by tasks that manipulate the final reward.

        :param state: current state
        :param remaining_steps: number of time steps left in the episode
        :return: final reward
        """
        return 0.

    @abstractmethod
    def has_succeeded(self, state: np.ndarray) -> bool:
        """
        Check the environment if the agent succeeded.

        :param state: environments current state
        :return: `True` if succeeded
        """
        raise NotImplementedError

    def has_failed(self, state: np.ndarray) -> bool:
        """
        Check the environment if the agent failed.
        The default implementation checks if the state is out of bounds.

        :param state: environments current state
        :return: `True` if failed
        """
        return not self.env_spec.state_space.contains(state)

    def is_done(self, state: np.ndarray) -> bool:
        """
        Check if a final state is reached.

        .. note::
            It is strongly recommended to call this method every environment step.

        :param state: current state
        :param act: current action
        :return done: done flag
        """
        return self.has_succeeded(state) or self.has_failed(state)


def all_tasks(task):
    """
    Iterates over the task chain.

    :param task: outermost task of the chain
    :return: an iterable over the whole chain from outermost to innermost
    """
    yield task  # outermost
    while isinstance(task, TaskWrapper):
        task = task.wrapped_task
        yield task


class TaskWrapper(Task):
    """ Base for all task wrappers. Delegates all environment methods to the wrapped environment. """

    def __init__(self, wrapped_task: Task):
        """
        Constructor

        :param wrapped_task: task to wrap
        """
        if not isinstance(wrapped_task, Task):
            raise pyrado.TypeErr(given=wrapped_task, expected_type=Task)

        self._wrapped_task = wrapped_task

    @property
    def env_spec(self) -> EnvSpec:
        return self._wrapped_task.env_spec

    @property
    def wrapped_task(self):
        return self._wrapped_task

    @property
    def state_des(self) -> np.ndarray:
        return self._wrapped_task.state_des

    @state_des.setter
    def state_des(self, state_des: np.ndarray):
        self._wrapped_task.state_des = state_des

    @property
    def space_des(self) -> Space:
        return self._wrapped_task.space_des

    @space_des.setter
    def space_des(self, space_des: Space):
        self._wrapped_task.space_des = space_des

    @property
    def rew_fcn(self) -> RewFcn:
        return self._wrapped_task.rew_fcn

    def reset(self, **kwargs):
        return self._wrapped_task.reset(**kwargs)

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        return self._wrapped_task.step_rew(state, act, remaining_steps)

    def has_succeeded(self, state: np.ndarray) -> bool:
        return self._wrapped_task.has_succeeded(state)

    def has_failed(self, state: np.ndarray) -> bool:
        return self._wrapped_task.has_failed(state)

    def is_done(self, state: np.ndarray) -> bool:
        return self._wrapped_task.is_done(state)
