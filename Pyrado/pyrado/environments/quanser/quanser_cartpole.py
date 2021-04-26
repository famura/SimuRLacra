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

import time
from abc import abstractmethod
from typing import Optional

import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.quanser import max_act_qcp
from pyrado.environments.quanser.base import QuanserReal
from pyrado.policies.special.environment_specific import QCartPoleGoToLimCtrl
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.compound import CompoundSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.final_reward import FinalRewMode, FinalRewTask
from pyrado.tasks.reward_functions import QuadrErrRewFcn, UnderActuatedSwingUpRewFcn
from pyrado.utils.input_output import completion_context, print_cbt


class QCartPoleReal(QuanserReal, Serializable):
    """Base class for the real Quanser Cart-Pole"""

    def __init__(self, dt: float, max_steps: int, task_args: Optional[dict] = None, ip: Optional[str] = "192.168.2.38"):
        """
        Constructor

        :param dt: time step size on the Quanser device [s]
        :param max_steps: maximum number of steps executed on the device
        :param task_args: arguments for the task construction
        :param ip: IP address of the Cart-Pole platform
        """
        Serializable._init(self, locals())

        # Initialize spaces, dt, max_step, and communication
        super().__init__(ip, rcv_dim=4, snd_dim=1, dt=dt, max_steps=max_steps, task_args=task_args)
        self._curr_act = np.zeros(self.act_space.shape)  # just for usage in render function

        # Calibration and limits
        self._l_rail = 0.814  # [m]
        self._x_buffer = 0.05  # [m]
        self._calibrated = False
        self._c_lim = 0.075
        self._norm_x_lim = np.zeros(2)

    def _create_spaces(self):
        # Define the spaces
        self._state_space = None  # needs to be set in subclasses
        max_obs = np.array([0.814 / 2.0, 1.0, 1.0, pyrado.inf, pyrado.inf])
        self._obs_space = BoxSpace(-max_obs, max_obs, labels=["x", "sin_theta", "cos_theta", "x_dot", "theta_dot"])
        self._act_space = BoxSpace(-max_act_qcp, max_act_qcp, labels=["V"])

    @abstractmethod
    def _create_task(self, task_args: dict):
        raise NotImplementedError

    @property
    def task(self):
        return self._task

    def observe(self, state) -> np.ndarray:
        return np.array([state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    def calibrate(self):
        if self._calibrated:
            return
        print_cbt("Calibrating the cart-pole ...", "c")

        # Go to the left
        with completion_context("Going to the left", color="c", bright=True):
            obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
            ctrl = QCartPoleGoToLimCtrl(obs, positive=True)
            while not ctrl.done:
                act = ctrl(obs)
                obs, _, _, _ = self.step(act)

            if ctrl.success:
                self._norm_x_lim[1] = obs[0]
            else:
                raise RuntimeError("Going to the left limit failed!")

        # Go to the right
        with completion_context("Going to the right", color="c", bright=True):
            obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
            ctrl = QCartPoleGoToLimCtrl(obs, positive=False)
            while not ctrl.done:
                act = ctrl(obs)
                obs, _, _, _ = self.step(act)

            if ctrl.success:
                self._norm_x_lim[0] = obs[0]
            else:
                raise RuntimeError("Going to the right limit failed!")

        # Activate the absolute cart position:
        self._calibrated = True

    def _center_cart(self):
        """Move the cart to the center (x = 0)."""
        # Initialize
        t_max, t_start = 10.0, time.time()
        obs, _, _, _ = self.step(np.zeros(self.act_space.shape))

        with completion_context("Centering the cart", color="c", bright=True):
            while (time.time() - t_start) < t_max:
                act = -np.sign(obs[0]) * 1.5 * np.ones(self.act_space.shape)
                obs, _, _, _ = self.step(act)

                if np.abs(obs[0]) <= self._c_lim / 10.0:
                    break

            # Stop the cart
            obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
            time.sleep(0.5)

            if np.abs(obs[0]) > self._c_lim:
                # time.sleep(0.1)
                raise RuntimeError(f"Centering of the cart failed: |x| = {np.abs(obs[0]):.2f} > {self._c_lim:.2f}")

    def _correct_sensor_offset(self, meas: np.ndarray) -> np.ndarray:
        # Transform the relative cart position to [-0.4, +0.4]
        if self._calibrated:
            meas[0] = (meas[0] - self._norm_x_lim[0]) - 0.5 * (self._norm_x_lim[1] - self._norm_x_lim[0])

        # Normalize the angle from -pi to +pi:
        meas[1] = np.mod(meas[1] + np.pi, 2 * np.pi) - np.pi

        return meas


class QCartPoleStabReal(QCartPoleReal):
    """Stabilization task on the real Quanser Cart-Pole"""

    name: str = "qcp-st"

    def __init__(
        self,
        dt: Optional[float] = 1 / 500.0,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        ip: Optional[str] = "192.168.2.38",
    ):
        """
        Constructor

        :param dt: time step size on the Quanser device [s]
        :param max_steps: maximum number of steps executed on the device
        :param task_args: arguments for the task construction
        :param ip: IP address of the Cart-pole platform
        """
        super().__init__(dt, max_steps, task_args, ip)

        # Define the task-specific state space
        stab_thold = 15 / 180.0 * np.pi  # threshold angle for the stabilization task to be a failure [rad]
        min_state_1 = np.array([-self._l_rail / 2.0 + self._x_buffer, np.pi - stab_thold, -np.inf, -np.inf])
        max_state_1 = np.array([+self._l_rail / 2.0 - self._x_buffer, np.pi + stab_thold, np.inf, np.inf])
        min_state_2 = np.array([-self._l_rail / 2.0 + self._x_buffer, -np.pi - stab_thold, -np.inf, -np.inf])
        max_state_2 = np.array([+self._l_rail / 2.0 - self._x_buffer, -np.pi + stab_thold, np.inf, np.inf])

        bs_1 = BoxSpace(min_state_1, max_state_1, labels=["x", "theta", "x_dot", "theta_dot"])
        bs_2 = BoxSpace(min_state_2, max_state_2, labels=["x", "theta", "x_dot", "theta_dot"])

        self._state_space = CompoundSpace([bs_1, bs_2])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([5e-0, 1e1, 1e-2, 1e-2]))
        R = task_args.get("R", np.diag([1e-3]))

        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, QuadrErrRewFcn(Q, R), idcs=[1]),
            mode=FinalRewMode(state_dependent=True, time_dependent=True),
        )

    def _wait_for_upright_pole(self):
        """Waiting until the user manually set the pole upright"""
        with completion_context("Centering the cart", color="c", bright=True):
            # Initialize
            t_max, t_start = 15.0, time.time()
            upright = False

            pos_th = np.array([self._c_lim, 2.0 * np.pi / 180.0])
            vel_th = 0.1 * np.ones(2)
            th = np.hstack((pos_th, vel_th))

            # Wait until the pole is upright
            while (time.time() - t_start) <= t_max:
                obs, _, _, _ = self.step(np.zeros(self.act_space.shape))
                time.sleep(1 / 550.0)

                abs_err = np.abs(np.array([obs[1], obs[2]]) - np.array([[0.0, -1.0]]))
                err_th = np.array([np.sin(np.deg2rad(3.0)), np.sin(np.deg2rad(3.0))])

                if np.all(abs_err <= err_th):
                    upright = True
                    break

            if not upright:
                # time.sleep(0.1)
                state_str = np.array2string(
                    np.abs(obs),
                    suppress_small=True,
                    precision=2,
                    formatter={"float_kind": lambda x: "{0:+05.2f}".format(x)},
                )
                th_str = np.array2string(
                    th, suppress_small=True, precision=2, formatter={"float_kind": lambda x: "{0:+05.2f}".format(x)}
                )
                raise TimeoutError("The pole is not upright: {0} > {1}".format(state_str, th_str))

    def reset(self, *args, **kwargs):
        # Reset socket, task, and calibrate
        super().reset(args, kwargs)

        # The system only needs to be calibrated once, as this is a bit time consuming
        self.calibrate()

        # Center the cart in the middle
        self._center_cart()

        # Wait until the human reset the pole properly
        self._wait_for_upright_pole()

        # Start with a zero action and get the first sensor measurements
        self.state = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Reset time counter
        self._curr_step = 0

        return self.observe(self.state)


class QCartPoleSwingUpReal(QCartPoleReal):
    """Swing-up task on the real Quanser Cart-Pole"""

    name: str = "qcp-su"

    def __init__(
        self,
        dt: Optional[float] = 1 / 500.0,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        ip: Optional[str] = "192.168.2.38",
    ):
        """
        Constructor

        :param dt: time step size on the Quanser device [s]
        :param max_steps: maximum number of steps executed on the device
        :param task_args: arguments for the task construction
        :param ip: IP address of the Cart-pole platform
        """
        super().__init__(dt, max_steps, task_args, ip)

        # Define the task-specific state space
        max_state = np.array([self._l_rail / 2.0 - self._x_buffer, +4 * np.pi, np.inf, np.inf])  # [m, rad, m/s, rad/s]
        min_state = np.array(
            [-self._l_rail / 2.0 + self._x_buffer, -4 * np.pi, -np.inf, -np.inf]
        )  # [m, rad, m/s, rad/s]
        self._state_space = BoxSpace(min_state, max_state, labels=["x", "theta", "x_dot", "theta_dot"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))

        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, UnderActuatedSwingUpRewFcn(), idcs=[1]),
            mode=FinalRewMode(always_negative=True),
        )

    def reset(self, *args, **kwargs):
        # Reset socket and task
        super().reset()

        # The system only needs to be calibrated once, as this is a bit time consuming
        self.calibrate()

        # Center the cart in the middle
        self._center_cart()

        # Start with a zero action and get the first sensor measurements
        meas = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))
        self.state = meas

        # Reset time counter
        self._curr_step = 0

        return self.observe(self.state)
