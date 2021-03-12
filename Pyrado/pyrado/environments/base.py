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
from abc import ABC, abstractmethod
from colorama import Style
from init_args_serializer import Serializable
from typing import Optional, Union

import pyrado
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils import get_class_name
from pyrado.utils.data_types import EnvSpec, RenderMode


class Env(ABC, Serializable):
    """ Base class of all environments in Pyrado. Uses Serializable to facilitate proper serialization. """

    name: str = None  # unique identifier

    def __init__(self, dt: Union[int, float], max_steps: Union[int, float] = pyrado.inf):
        """
        Constructor

        :param dt: integration step size in seconds, default value is used for for one-step environments
        :param max_steps: max number of simulation time steps
        """
        if not isinstance(dt, (int, float)):
            raise pyrado.TypeErr(given=dt, expected_type=(int, float))
        if dt < 0:
            raise pyrado.ValueErr(given=dt, ge_constraint="0")
        if max_steps < 1:
            raise pyrado.ValueErr(given=max_steps, ge_constraint="1")
        self._dt = float(dt)
        self._max_steps = max_steps
        self._curr_step = 0
        self._curr_rew = -pyrado.inf  # only for initialization
        self.state = None  # current state of the environment

    def __str__(self):
        """ Get an information string. """
        return (
            Style.BRIGHT
            + f"{get_class_name(self)}"
            + Style.RESET_ALL
            + f" (id: {id(self)})\n"
            + Style.BRIGHT
            + "state space: "
            + Style.RESET_ALL
            + f"\n{str(self.state_space)}\n"
            + Style.BRIGHT
            + "observation space: "
            + Style.RESET_ALL
            + f"\n{str(self.obs_space)}\n"
            + Style.BRIGHT
            + "action space: "
            + Style.RESET_ALL
            + f"\n{str(self.act_space)}\n"
        )

    @property
    @abstractmethod
    def state_space(self) -> Space:
        """ Get the space of the states (used for describing the environment). """
        raise NotImplementedError

    @property
    @abstractmethod
    def obs_space(self) -> Space:
        """ Get the space of the observations (agent's perception of the environment). """
        raise NotImplementedError

    @property
    @abstractmethod
    def act_space(self) -> Space:
        """ Get the space of the actions. """
        raise NotImplementedError

    @property
    def spec(self) -> EnvSpec:
        """ Get the environment specification (generated on call). """
        return EnvSpec(self.obs_space, self.act_space, self.state_space)

    @property
    def dt(self) -> float:
        """ Get the time step size. """
        return self._dt

    @dt.setter
    def dt(self, dt: Union[int, float]):
        """ Set the time step size. """
        if not dt > 0:
            raise pyrado.ValueErr(given=dt, g_constraint="0")
        if not isinstance(dt, (float, int)):
            raise pyrado.TypeErr(given=dt, expected_type=[float, int])
        self._dt = float(dt)

    @property
    def max_steps(self) -> Union[int, float]:
        """
        Get the maximum number of simulation steps.

        .. note::
            The step count should always be an integer. Some environments have no maximum step size. For these,
            `float('Inf')` should be used, since it is the only value larger then any int.

        :return: maximum number of time steps before the environment terminates
        """
        return self._max_steps

    @max_steps.setter
    def max_steps(self, num_steps: Union[int, float]):
        """
        Set the maximal number of dynamic steps in the environment

        :param num_steps: number of steps
        """
        if not (isinstance(num_steps, int) or num_steps == pyrado.inf):
            raise pyrado.TypeErr(msg=f"Number of steps needs to be an integer of infinite, but is {num_steps}")
        if not num_steps > 0:
            raise pyrado.ValueErr(given=num_steps, g_constraint="0")
        self._max_steps = num_steps

    @property
    def curr_step(self) -> int:
        """ Get the number of the current simulation step (0 for the initial step). """
        return self._curr_step

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        """
        Create task based on the domain parameters and spaces.

        .. note::
            This function should be called from the environment's constructor.

        :param task_args: arguments for the task construction, e.g `dict(state_des=np.zeros(42))`
        :return: task of the environment
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def task(self) -> Task:
        """ Get the task describing what the agent should do in the environment. """
        raise NotImplementedError

    @abstractmethod
    def reset(self, init_state: Optional[np.ndarray] = None, domain_param: Optional[dict] = None) -> np.ndarray:
        """
        Reset the environment to its initial state and optionally set different domain parameters.

        :param init_state: set explicit initial state if not None. Must match `init_space` if any.
        :param domain_param: set explicit domain parameters if not None
        :return obs: initial observation of the state.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, act: np.ndarray) -> tuple:
        """
        Perform one time step of the simulation or on the real-world device.
        When a terminal condition is met, the reset function is called.

        .. note::
            This function is responsible for limiting the actions, i.e. has to call `limit_act()`.

        :param act: action to be taken in the step
        :return obs: current observation of the environment
        :return reward: reward depending on the selected reward function
        :return done: indicates whether the episode has ended
        :return env_info: contains diagnostic information about the environment
        """
        raise NotImplementedError

    def observe(self, state: np.ndarray) -> np.ndarray:
        """
        Compute the (noise-free) observation from the current state.

        .. note::
            This method should be overwritten if the environment has a distinct observation space.

        :param state: current state of the environment
        :return: observation perceived to the agent
        """
        return state.copy()

    def limit_act(self, act: np.ndarray) -> np.ndarray:
        """
        Clip the actions according to the environment's action space. Note, this also affects the exploration.

        :param act: unbounded action
        :return: bounded action
        """
        return self.act_space.project_to(act)

    @abstractmethod
    def render(self, mode: RenderMode, render_step: Optional[int] = 1):
        """
        Visualize one time step.

        :param mode: render mode: console, video, or both
        :param render_step: interval for rendering
        """
        raise NotImplementedError

    def close(self):
        """ Disconnect from the device. """
        raise NotImplementedError
