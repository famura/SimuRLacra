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

from typing import NamedTuple

import numpy as np
from colorama import Style
from tabulate import tabulate

import pyrado
from pyrado.tasks.base import Task, TaskWrapper
from pyrado.utils import get_class_name
from pyrado.utils.input_output import print_cbt


class FinalRewMode(NamedTuple):
    """The specification of how the final state should be rewarded or punished"""

    state_dependent: bool = False
    time_dependent: bool = False
    always_positive: bool = False
    always_negative: bool = False
    user_input: bool = False

    def __str__(self):
        """Get an information string."""
        return (
            Style.BRIGHT
            + f"{get_class_name(self)}"
            + Style.RESET_ALL
            + f" (id: {id(self)})\n"
            + tabulate(
                [
                    ["state_dependent", self.state_dependent],
                    ["time_dependent", self.time_dependent],
                    ["always_positive", self.always_positive],
                    ["always_negative", self.always_negative],
                    ["user_input", self.user_input],
                ]
            )
        )


class FinalRewTask(TaskWrapper):
    """
    Wrapper for tasks which yields a reward / cost on success / failure

    :usage:
    .. code-block:: python

        task = FinalRewTask(DesStateTask(spec, state_des, rew_fcn, success_fcn), mode=FinalRewMode(), factor=1e3)
    """

    def __init__(self, wrapped_task: Task, mode: FinalRewMode, factor: float = 1e3):
        """
        Constructor

        :param wrapped_task: task to wrap
        :param mode: mode for calculating the final reward
        :param factor: (positive) value to scale the final reward.
                       The `factor` is ignored if `mode.time_dependent` is `True`
        """
        # Call TaskWrapper's constructor
        super().__init__(wrapped_task)

        if not isinstance(mode, FinalRewMode):
            raise pyrado.TypeErr(given=mode, expected_type=FinalRewMode)
        if mode.user_input and (
            mode.always_positive or mode.always_negative or mode.state_dependent or mode.time_dependent
        ):
            print_cbt("If the user_input == True, then all other specifications in FinalRewMode are ignored.", "w")

        self.mode = mode
        self.factor = abs(factor)
        self._yielded_final_rew = False

    @property
    def yielded_final_rew(self) -> bool:
        """Get the flag that signals if this instance already yielded its final reward."""
        return self._yielded_final_rew

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._yielded_final_rew = False

    def compute_final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the reward / cost on task completion / fail of this task.

        :param state: current state of the environment
        :param remaining_steps: number of time steps left in the episode
        :return: final reward of this task
        """

        def mode_switch(on=(), off=()):
            """
            Helper function to avoid complex logical expressions.

            :param on: all values that have to be true
            :param off: all values that have to be false
            :return: result
            """
            return all(on) and not any(off)

        if self._yielded_final_rew:
            # Only yield the final reward once
            return 0.0

        else:
            self._yielded_final_rew = True

            if not self.mode.user_input:
                # Default case
                if mode_switch(
                    off=(
                        self.mode.always_positive,
                        self.mode.always_negative,
                        self.mode.state_dependent,
                        self.mode.time_dependent,
                    )
                ):
                    if self.has_failed(state):
                        return -1.0 * self.factor
                    elif self.has_succeeded(state):
                        return self.factor
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.always_positive,),
                    off=(self.mode.always_negative, self.mode.state_dependent, self.mode.time_dependent),
                ):
                    if self.has_failed(state):
                        return 0.0
                    elif self.has_succeeded(state):
                        return self.factor
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.always_negative,),
                    off=(self.mode.always_positive, self.mode.state_dependent, self.mode.time_dependent),
                ):
                    if self.has_failed(state):
                        return -1.0 * self.factor
                    elif self.has_succeeded(state):
                        return 0.0
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.always_positive, self.mode.state_dependent),
                    off=(self.mode.always_negative, self.mode.time_dependent),
                ):
                    if self.has_failed(state):
                        return 0.0
                    elif self.has_succeeded(state):
                        act = np.zeros(self.env_spec.act_space.shape)  # dummy
                        step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                        return self.factor * abs(step_rew)
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.always_negative, self.mode.state_dependent),
                    off=(self.mode.always_positive, self.mode.time_dependent),
                ):
                    if self.has_failed(state):
                        act = np.zeros(self.env_spec.act_space.shape)  # dummy
                        step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                        return -1.0 * self.factor * abs(step_rew)
                    elif self.has_succeeded(state):
                        return 0.0
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.state_dependent,),
                    off=(self.mode.always_positive, self.mode.always_negative, self.mode.time_dependent),
                ):
                    act = np.zeros(self.env_spec.act_space.shape)  # dummy
                    step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                    if self.has_failed(state):
                        return -1.0 * self.factor * abs(step_rew)
                    elif self.has_succeeded(state):
                        return self.factor * abs(step_rew)
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.state_dependent, self.mode.time_dependent),
                    off=(self.mode.always_positive, self.mode.always_negative),
                ):
                    act = np.zeros(self.env_spec.act_space.shape)  # dummy
                    step_rew = self._wrapped_task.step_rew(state, act, remaining_steps)
                    if self.has_failed(state):
                        return -1.0 * remaining_steps * abs(step_rew)
                    elif self.has_succeeded(state):
                        return remaining_steps * abs(step_rew)
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.time_dependent,),
                    off=(self.mode.always_positive, self.mode.always_negative, self.mode.state_dependent),
                ):
                    if self.has_failed(state):
                        return -1.0 * remaining_steps
                    elif self.has_succeeded(state):
                        return remaining_steps
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.always_positive, self.mode.time_dependent),
                    off=(self.mode.always_negative, self.mode.state_dependent),
                ):
                    if self.has_failed(state):
                        return 0.0
                    elif self.has_succeeded(state):
                        return remaining_steps
                    else:
                        return 0.0

                elif mode_switch(
                    on=(self.mode.always_negative, self.mode.time_dependent),
                    off=(self.mode.always_positive, self.mode.state_dependent),
                ):
                    if self.has_failed(state):
                        return -1.0 * remaining_steps
                    elif self.has_succeeded(state):
                        return 0.0
                    else:
                        return 0.0

                else:
                    raise NotImplementedError(
                        f"No matching configuration found for the given FinalRewMode:\n{self.mode}"
                    )

            else:
                user_rew = None
                while user_rew is None:
                    user_input = input("Please enter a final reward: ")
                    try:
                        user_rew = float(user_input)
                    except ValueError:
                        print_cbt("The received input could not be casted to a float. Try again: ", "y")
                        user_rew = None
                return user_rew


class BestStateFinalRewTask(TaskWrapper):
    """
    Wrapper for tasks which yields a reward / cost on success / failure based on the best reward / cost observed in the
    current trajectory.
    """

    def __init__(self, wrapped_task: Task, factor: float):
        """
        Constructor

        :param wrapped_task: task to wrap
        :param factor: value to scale the final reward
        """
        # Call TaskWrapper's constructor
        super().__init__(wrapped_task)

        self.factor = factor
        self.best_rew = -pyrado.inf
        self._yielded_final_rew = False

    @property
    def yielded_final_rew(self) -> bool:
        """Get the flag that signals if this instance already yielded its final reward."""
        return self._yielded_final_rew

    def step_rew(self, state: np.ndarray, act: np.ndarray, remaining_steps: int) -> float:
        rew = self._wrapped_task.step_rew(state, act, remaining_steps)
        if rew > self.best_rew:
            self.best_rew = rew
        return rew

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.best_rew = -pyrado.inf
        self._yielded_final_rew = False

    def compute_final_rew(self, state: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the reward / cost on task completion / fail of this task.

        :param state: current state of the environment
        :param remaining_steps: number of time steps left in the episode
        :return: final reward of this task
        """
        if self._yielded_final_rew:
            # Only yield the final reward once
            return 0.0
        else:
            self._yielded_final_rew = True

            # Return the highest reward / lowest cost scaled with the factor
            return self.best_rew * self.factor
