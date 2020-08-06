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

import pyrado
from pyrado.environments.quanser.quanser_common import QSocket
from pyrado.environments.real_base import RealEnv
from pyrado.spaces.base import Space
from pyrado.tasks.base import Task
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


class QuanserReal(RealEnv, ABC):
    """ Base class of all real-world Quanser environments in Pyrado """

    def __init__(self,
                 ip: str,
                 rcv_dim: int,
                 snd_dim: int,
                 dt: float = 1/500.,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None):
        """
        Constructor

        :param ip: IP address of the platform
        :param rcv_dim: number of dimensions of the sensor i.e. measurement signal (received from Simulink server)
        :param snd_dim: number of dimensions of the action command (send to Simulink server)
        :param dt: sampling time interval
        :param max_steps: maximum number of time steps
        :param task_args: arguments for the task construction
        """
        # Call the base class constructor to initialize fundamental members
        super().__init__(dt, max_steps)

        # Initialize the state since it is needed for the first time the step fcn is called (in the reset fcn)
        self.state = np.zeros(rcv_dim)
        self._curr_act = None  # just for usage in render function

        # Create a socket for communicating with the Quanser devices
        self._qsoc = QSocket(ip, rcv_dim, snd_dim)

        # Initialize spaces
        self._state_space = None
        self._obs_space = None
        self._act_space = None
        self._create_spaces()

        # Create task
        if not (isinstance(task_args, dict) or task_args is None):
            raise pyrado.TypeErr(given=task_args, expected_type=dict)
        self._task = self._create_task(task_args=dict() if task_args is None else task_args)

    def __del__(self):
        """ Finalizer forwards to close function. """
        self.close()

    def close(self):
        """ Sends a zero-step and closes the communication. """
        if self._qsoc.is_open():
            self.step(np.zeros(self.act_space.shape))
            self._qsoc.close()
            print_cbt('Closed the connection to the Quanser device.', 'c', bright=True)

    @property
    def state_space(self) -> Space:
        return self._state_space

    @property
    def obs_space(self) -> Space:
        return self._obs_space

    @property
    def act_space(self) -> Space:
        return self._act_space

    @property
    def task(self) -> Task:
        return self._task

    @abstractmethod
    def _create_spaces(self):
        """
        Create spaces based on the domain parameters.
        Should set the attributes `_state_space`, `_obs_space`, and `_act_space`.

        .. note::
            This function is called from the constructor.
        """

    @abstractmethod
    def reset(self, *args, **kwargs):
        """
        Reset the environment.
        The base version (re-)opens the socket and resets the task.

        :param args: just for compatibility with SimEnv. All args can be ignored.
        :param kwargs: just for compatibility with SimEnv. All kwargs can be ignored.
        """
        # Cancel and re-open the connection to the socket
        self._qsoc.close()
        self._qsoc.open()
        print_cbt('Opened the connection to the Quanser device.', 'c', bright=True)

        # Reset the task
        self._task.reset(env_spec=self.spec)

    def _correct_sensor_offset(self, meas: np.ndarray) -> np.ndarray:
        """
        Correct the sensor's offset. Does nothing by default.

        :param meas: raw measurements from the device
        :return: corrected measurements
        """
        return meas

    def step(self, act):
        info = dict(t=self._curr_step*self._dt, act_raw=act)

        # Current reward depending on the (measurable) state and the current (unlimited) action
        remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
        self._curr_rew = self._task.step_rew(self.state, act, remaining_steps)

        # Apply actuator limits
        act_lim = self.limit_act(act)
        self._curr_act = act_lim

        # Send actions and receive sensor measurements
        meas = self._qsoc.snd_rcv(act_lim)

        # Correct for offset, and construct the state from the measurements
        meas = self._correct_sensor_offset(meas)
        self.state = meas
        self._curr_step += 1

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        # Add final reward if done
        if done:
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, info

    def render(self, mode: RenderMode = RenderMode(text=True), render_step: int = 1):
        """
        Visualize one time step of the real-world device.
        The base version prints to console when the state exceeds its boundaries.

        :param mode: render mode: console, video, or both
        :param render_step: interval for rendering
        """
        if self._curr_step%render_step == 0:
            if mode.text:
                print(f'step: {self._curr_step:4d}  |  '
                      f'in bounds: {self._state_space.contains(self.state):1d}  |  '
                      f'rew: {self._curr_rew:1.3f}  |  '
                      f'act: {self._curr_act}  |  '
                      f'next state: {self.state}')
            if mode:
                # Print out of bounds to console if the mode is not empty
                self.state_space.contains(self.state, verbose=True)
