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

from typing import Optional

import numpy as np
import torch as to
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.quanser import max_act_qq
from pyrado.environments.quanser.base import QuanserReal
from pyrado.policies.special.environment_specific import QQubeGoToLimCtrl, QQubePDCtrl
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn
from pyrado.utils.input_output import completion_context, print_cbt


class QQubeSwingUpReal(QuanserReal, Serializable):
    """Class for the real Quanser Qube a.k.a. Furuta pendulum"""

    name: str = "qq-su"

    def __init__(
        self,
        dt: float = 1 / 500.0,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        ip: str = "192.168.2.17",
    ):
        """
        Constructor

        :param dt: sampling frequency on the Quanser device [Hz]
        :param max_steps: maximum number of steps executed on the device [-]
        :param task_args: arguments for the task construction
        :param ip: IP address of the Qube platform
        """
        Serializable._init(self, locals())

        # Initialize spaces, dt, max_step, and communication
        super().__init__(ip, rcv_dim=4, snd_dim=1, dt=dt, max_steps=max_steps, task_args=task_args)
        self._curr_act = np.zeros(self.act_space.shape)  # just for usage in render function
        self._sens_offset = np.zeros(4)  # last two entries are never calibrated but useful for broadcasting

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([3e-1, 1.0, 2e-2, 5e-3]))
        R = task_args.get("R", np.diag([4e-3]))

        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([120.0 / 180 * np.pi, 4 * np.pi, 20 * np.pi, 20 * np.pi])  # [rad, rad, rad/s, rad/s]
        max_obs = np.array([1.0, 1.0, 1.0, 1.0, pyrado.inf, pyrado.inf])  # [-, -, -, -, rad/s, rad/s]
        self._state_space = BoxSpace(-max_state, max_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])
        self._obs_space = BoxSpace(
            -max_obs, max_obs, labels=["sin_theta", "cos_theta", "sin_alpha", "cos_alpha", "theta_dot", "alpha_dot"]
        )
        self._act_space = BoxSpace(-max_act_qq, max_act_qq, labels=["V"])

    @property
    def task(self) -> Task:
        return self._task

    def observe(self, state) -> np.ndarray:
        return np.array([np.sin(state[0]), np.cos(state[0]), np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    def reset(self, *args, **kwargs) -> np.ndarray:
        # Reset socket and task
        super().reset()

        # Run calibration routine to start in the center
        self.calibrate()

        # Start with a zero action and get the first sensor measurements
        meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Correct for offset, and construct the state from the measurements
        self.state = self._correct_sensor_offset(meas)

        # Reset time counter
        self._curr_step = 0

        return self.observe(self.state)

    def _correct_sensor_offset(self, meas: np.ndarray) -> np.ndarray:
        return meas - self._sens_offset

    def _wait_for_pole_at_rest(self, thold_ang_vel: float = 0.1 / 180.0 * np.pi):
        """Wait until the Qube's rotating pole is at rest"""
        cnt = 0
        while cnt < 1.5 / self._dt:
            # Get next measurement
            meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

            if np.abs(meas[2]) < thold_ang_vel and np.abs(meas[3]) < thold_ang_vel:
                cnt += 1
            else:
                cnt = 0

    def calibrate(self):
        """Calibration routine to move to the init position and determine the sensor offset"""
        with completion_context("Estimating sensor offset", color="c", bright=True):
            # Reset calibration
            self._sens_offset = np.zeros(4)  # last two entries are never calibrated but useful for broadcasting
            self._wait_for_pole_at_rest()

            # Create parts of the calibration controller
            go_right = QQubeGoToLimCtrl(positive=True, cnt_done=int(1.5 / self._dt))
            go_left = QQubeGoToLimCtrl(positive=False, cnt_done=int(1.5 / self._dt))
            go_center = QQubePDCtrl(self.spec)

            # Estimate alpha offset. Go to both limits for theta calibration.
            meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))
            while not go_right.done:
                meas = self._qsoc.snd_rcv(go_right(to.from_numpy(meas)))
            while not go_left.done:
                meas = self._qsoc.snd_rcv(go_left(to.from_numpy(meas)))
            self._sens_offset[0] = (go_right.th_lim + go_left.th_lim) / 2

            # Estimate alpha offset
            self._wait_for_pole_at_rest()
            meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))
            self._sens_offset[1] = meas[1]

        print_cbt(
            f"Sensor offset: "
            f"theta = {self._sens_offset[0]*180/np.pi:.3f} deg, "
            f"alpha = {self._sens_offset[1]*180/np.pi:.3f} deg",
            "g",
        )

        with completion_context("Centering cube", color="c", bright=True):
            meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))
            while not go_center.done:
                meas = self._qsoc.snd_rcv(go_center(to.from_numpy(meas - self._sens_offset)))
