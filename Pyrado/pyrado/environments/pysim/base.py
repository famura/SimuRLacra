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
from abc import abstractmethod
from copy import deepcopy
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.utils.data_types import RenderMode
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task


class SimPyEnv(SimEnv, Serializable):
    """ Base class for simulated environments implemented in pure Python """

    def __init__(self, dt: float, max_steps: int = pyrado.inf, task_args: [dict, None] = None):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        """
        Serializable._init(self, locals())
        super().__init__(dt, max_steps)

        # Initialize the domain parameters and the derived constants
        self._domain_param = self.get_nominal_domain_param()
        self._set_domain_param_attrs(self.get_nominal_domain_param())
        self._calc_constants()

        # Initialize spaces
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._init_space = None
        self._create_spaces()

        # Create task
        if not (isinstance(task_args, dict) or task_args is None):
            raise pyrado.TypeErr(given=task_args, expected_type=dict)
        self._task = self._create_task(task_args=dict() if task_args is None else task_args)

        # Animation with VPython
        self._curr_act = np.zeros(self.act_space.shape)
        self._anim = dict(canvas=None)

    @property
    def state_space(self) -> Space:
        return self._state_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def init_space(self) -> Space:
        return self._init_space

    @property
    def act_space(self) -> Space:
        return self._act_space

    @property
    def task(self) -> Task:
        return self._task

    @property
    def domain_param(self) -> dict:
        return deepcopy(self._domain_param)

    @domain_param.setter
    def domain_param(self, param: dict):
        if not isinstance(param, dict):
            raise pyrado.TypeErr(given=param, expected_type=dict)
        # Update the parameters
        self._domain_param.update(param)
        self._calc_constants()

        # Update spaces
        self._create_spaces()

        # Reset task to adapt for the potentially changed spaces
        self._task.reset(env_spec=self.spec)

    @abstractmethod
    def _create_spaces(self):
        """
        Create spaces based on the domain parameters.
        Should set the attributes `_state_space`, `_act_space`, `_obs_space`, and `_init_space`.

        .. note::
            This function is called from the constructor and from the domain parameter setter.
        """
        raise NotImplementedError

    @abstractmethod
    def _step_dynamics(self, act: np.ndarray):
        """
        Implement this to apply the given action to the environment's current state.

        :param act: action
        """
        raise NotImplementedError

    def _calc_constants(self):
        """
        Called to calculate the physics constants that depend on the domain parameters. Override in subclasses.

        .. note::
            This function is called from the constructor and from the domain parameter setter.
        """
        pass

    def _set_domain_param_attrs(self, domain_param: dict):
        """
        Set all key value pairs of the given domain parameter dict to the state dict (i.e. make them an attribute).

        :param domain_param: dict of domain parameters to save a private attributes
        """
        for name in self.supported_domain_param:
            dp = domain_param.get(name, None)
            if dp is not None:
                setattr(self, name, dp)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset time
        self._curr_step = 0

        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param

        if init_state is None:
            # Sample init state from init state space
            init_state = self.init_space.sample_uniform()
        elif not isinstance(init_state, np.ndarray):
            # Make sure init state is a numpy array
            try:
                init_state = np.array(init_state)
            except Exception:
                raise pyrado.TypeErr(given=init_state, expected_type=[np.ndarray, list])

        if init_state.shape == self.state_space.shape:
            # Allow setting the complete state space
            if not self.state_space.contains(init_state, verbose=True):
                pyrado.ValueErr(msg="The full init state must be within the state space!")
            self.state = init_state.copy()
        else:
            # Set the initial state determined by an element of the init space
            if not self.init_space.contains(init_state, verbose=True):
                pyrado.ValueErr(msg="The init state must be within init state space!")
            self.state = self._state_from_init(init_state)

        # Reset the task
        self._task.reset(env_spec=self.spec)

        # Reset VPython animation
        if self._anim["canvas"] is not None:
            self._reset_anim()

        # Return an observation
        return self.observe(self.state)

    def _state_from_init(self, init_state: np.ndarray):
        """
        Expand the init state parameter into the full state vector.

        :param init_state: init state from init space
        :return: internal state
        """
        assert (
            self._init_space.shape == self._state_space.shape
        ), "Must override _state_from_init if init state space differs from state space!"
        return init_state

    def step(self, act: np.ndarray) -> tuple:
        # Current reward depending on the state (before step) and the (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self.task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act = self.limit_act(act)
        self._curr_act = act  # just for the render function

        # Apply the action and simulate the resulting dynamics
        self._step_dynamics(act)

        info = dict(t=self._curr_step * self._dt)
        self._curr_step += 1

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            # Add final reward if done
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def render(self, mode: RenderMode, render_step: int = 1):
        if self._curr_step % render_step == 0:
            # Call base class
            super().render(mode)

            # Print to console
            if mode.text:
                print(
                    "step: {:3}  |  r_t: {: 1.3f}  |  a_t: {}\t |  s_t+1: {}".format(
                        self._curr_step, self._curr_rew, self._curr_act, self.state
                    )
                )

            # VPython
            if mode.video:
                if self._anim["canvas"] is None:
                    self._init_anim()

                # Update the animation
                self._update_anim()

    def _init_anim(self):
        """
        Initialize animation. Called by first render call.
        :return:
        """
        pass

    def _update_anim(self):
        """
        Update animation. Called by each render call.
        :return:
        """
        pass

    def _reset_anim(self):
        """
        Reset animation to initial state. Called by reset() if needed.
        The default implementation does nothing.
        """
        pass
