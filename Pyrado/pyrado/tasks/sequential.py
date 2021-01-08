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
from copy import deepcopy
from typing import Sequence

import pyrado
from pyrado.spaces.base import Space
from pyrado.utils.data_types import EnvSpec
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import RewFcn
from pyrado.utils.input_output import print_cbt


class SequentialTasks(Task):
    """ Task class for a sequence of tasks a.k.a. goals """

    def __init__(
        self, tasks: Sequence[Task], start_idx: int = 0, hold_rew_when_done: bool = False, verbose: bool = False
    ):
        """
        Constructor

        :param tasks: sequence of tasks a.k.a. goals, the order matters
        :param start_idx: index of the task to start with, by default with the first one in the list
        :param hold_rew_when_done: if `True` reward values for done tasks will be stored and added every step
        :param verbose: print messages on task completion

        .. note::
            `hold_rew_when_done=True` only makes sense for positive rewards.
        """
        self._tasks = tasks
        self._idx_curr = start_idx
        self.succeeded_tasks = np.full(len(self), False, dtype=bool)
        self.failed_tasks = np.full(len(self), False, dtype=bool)
        self.succeeded_tasks[:start_idx] = True  # check off tasks which are before the start task
        self.hold_rew_when_done = hold_rew_when_done
        if self.hold_rew_when_done:
            self.held_rews = np.zeros(len(self))
        self.verbose = verbose

    def __len__(self) -> int:
        return len(self._tasks)

    @property
    def env_spec(self) -> EnvSpec:
        return self._tasks[0].env_spec  # safe to assume that all tasks have the same env_spec

    @property
    def tasks(self) -> Sequence[Task]:
        """ Get the list of tasks. """
        return deepcopy(self._tasks)

    @property
    def idx_curr(self) -> int:
        """ Get the index of the currently active task. """
        return self._idx_curr

    @idx_curr.setter
    def idx_curr(self, idx: int):
        """ Set the index of the currently active task. """
        if not (0 <= idx < len(self)):
            raise pyrado.ValueErr(given=idx, ge_constraint="0", le_constraint=f"{len(self) - 1}")
        self._idx_curr = idx

    @property
    def state_des(self) -> np.ndarray:
        """ Get the desired state the current task. """
        return self._tasks[self._idx_curr].state_des

    @state_des.setter
    def state_des(self, state_des: np.ndarray):
        """ Set the desired state the current task. """
        if not isinstance(state_des, np.ndarray):
            raise pyrado.TypeErr(given=state_des, expected_type=np.ndarray)
        self._tasks[self._idx_curr].state_des = state_des

    @property
    def space_des(self) -> Space:
        """ Get the desired space the current task. """
        return self._tasks[self._idx_curr].space_des

    @space_des.setter
    def space_des(self, space_des: Space):
        """ Set the desired space the current task. """
        if not isinstance(space_des, Space):
            raise pyrado.TypeErr(given=space_des, expected_type=Space)
        self._tasks[self._idx_curr].space_des = space_des

    @property
    def rew_fcn(self) -> RewFcn:
        """ Get the reward function of the current task. """
        return self._tasks[self._idx_curr].rew_fcn

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        """ Get the step reward from the current task. """
        step_rew = 0.0
        if self.hold_rew_when_done:
            for i in range(len(self)):
                # Iterate over previous tasks
                if self.succeeded_tasks[i] or self.failed_tasks[i]:
                    # Add the last reward from every done task (also true for failed tasks)
                    step_rew += self.held_rews[i]

        if not (self.succeeded_tasks[self._idx_curr] or self.failed_tasks[self._idx_curr]):
            # Only give step reward if current sub-task is active
            step_rew += self._tasks[self._idx_curr].step_rew(state, act, remaining_steps)

        final_rew = self._is_curr_task_done(state, act, remaining_steps)  # zero if the task is not done

        # self.logger.add_value('successful tasks', self.successful_tasks)
        return step_rew + final_rew

    def compute_final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the reward / cost on task completion / fail of this task.
        Since this task holds multiple sub-tasks, the final reward / cost is computed for them, too.

        .. note::
            The `ParallelTasks` class is not a subclass of `TaskWrapper`, i.e. this function only looks at the
            immediate sub-tasks.

        :param state: current state of the environment
        :param remaining_steps: number of time steps left in the episode
        :return: final reward of all sub-tasks
        """
        sum_final_rew = 0.0
        for t in self._tasks:
            sum_final_rew += t.compute_final_rew(state, remaining_steps)
        return sum_final_rew

    def reset(self, **kwargs):
        """ Reset all tasks. """
        self.idx_curr = 0
        for s in self._tasks:
            s.reset(**kwargs)

        # Reset internal check list for done tasks
        self.succeeded_tasks = np.full(len(self), False, dtype=bool)
        self.failed_tasks = np.full(len(self), False, dtype=bool)
        if "start_idx" in kwargs:
            self.succeeded_tasks[: kwargs["start_idx"]] = True

        # Reset the stored reward values for done tasks
        if self.hold_rew_when_done:
            self.held_rews = np.zeros(len(self))  # doesn't work with start_idx

    def _is_curr_task_done(
        self, state: np.ndarray, act: np.ndarray, remaining_steps: int, verbose: bool = False
    ) -> float:
        """
        Check if the current task is done. If so, move to the next one and return the final reward of this task.

        :param state: current state
        :param act: current action
        :param remaining_steps: number of time steps left in the episode
        :param verbose: print messages on success or failure
        :return: final return of the current subtask
        """
        if (
            not self.succeeded_tasks[self._idx_curr]
            and not self.failed_tasks[self._idx_curr]
            and self._tasks[self._idx_curr].is_done(state)
        ):
            # Task has not been marked done yet, but is now done

            if self._tasks[self._idx_curr].has_succeeded(state):
                # Check off successfully completed task
                self.succeeded_tasks[self._idx_curr] = True
                if verbose:
                    print_cbt(f"task {self._idx_curr} has succeeded (is done) at state {state}", "g")

            elif self._tasks[self._idx_curr].has_failed(state):
                # Check off unsuccessfully completed task
                self.failed_tasks[self._idx_curr] = True
                if verbose:
                    print_cbt(f"Task {self._idx_curr} has failed (is done) at state {state}", "r")

            else:
                raise pyrado.ValueErr(msg=f"Task {self._idx_curr} neither succeeded or failed but is done!")

            # Memorize current reward
            if self.hold_rew_when_done:
                self.held_rews[self._idx_curr] = self._tasks[self._idx_curr].step_rew(state, act, remaining_steps=0)

            # Give a reward for completing the task defined by the task
            task_final_rew = self._tasks[self._idx_curr].final_rew(state, remaining_steps)

            # Advance to the next task
            self.idx_curr = (self._idx_curr + 1) % len(self)

        else:
            task_final_rew = 0.0

        return task_final_rew

    def has_succeeded(self, state: np.ndarray) -> bool:
        """
        Check if this tasks is done. The SequentialTasks is successful if all sub-tasks are successful.

        :param state: environments current state
        :return: `True` if succeeded
        """
        successful = np.all(self.succeeded_tasks)
        if successful and self.verbose:
            print_cbt(f"All {len(self)} sequential sub-tasks are done successfully", "g")
        return successful
