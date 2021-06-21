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

import os.path as osp
from abc import abstractmethod
from typing import Optional

import mujoco_py
import numpy as np
import torch as to
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.mujoco.base import MujocoSimEnv
from pyrado.environments.quanser import max_act_qq
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn


class QQubeMjSim(MujocoSimEnv, Serializable):
    def __init__(
        self,
        frame_skip: int = 4,
        dt: Optional[float] = None,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
    ):
        """
        Constructor

        :param frame_skip: number of simulation frames for which the same action is held, results in a multiplier of
                           the time step size `dt`
        :param dt: by default the time step size is the one from the mujoco config file multiplied by the number of
                   frame skips (legacy from OpenAI environments). By passing an explicit `dt` value, this can be
                   overwritten. Possible use case if if you know that you recorded a trajectory with a specific `dt`.
        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction
        """
        Serializable._init(self, locals())
        model_path = osp.join(pyrado.MUJOCO_ASSETS_DIR, "furuta_pendulum.xml")
        super().__init__(model_path, frame_skip, dt, max_steps, task_args)
        self.camera_config = dict(distance=1.0, lookat=np.array((0.0, 0.0, 0.0)), elevation=-25.0, azimuth=180.0)

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        raise NotImplementedError

    @property
    def obs_space(self) -> Space:
        max_obs = np.array([1.0, 1.0, 1.0, 1.0, 20 * np.pi, 20 * np.pi])  # [-, -, -, -, rad/s, rad/s]
        return BoxSpace(
            -max_obs, max_obs, labels=["sin_theta", "cos_theta", "sin_alpha", "cos_alpha", "theta_dot", "alpha_dot"]
        )

    @property
    def act_space(self) -> Space:
        return BoxSpace(-max_act_qq, max_act_qq, labels=["V"])

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            g=9.81,  # gravity [m/s**2]
            Rm=8.4,  # motor resistance [Ohm]
            km=0.042,  # motor back-emf constant [V*s/rad]
            mass_rot_pole=0.095,  # rotary arm mass [kg]
            length_rot_pole=0.085,  # rotary arm length [m]
            damping_rot_pole=5e-6,  # rotary arm viscous damping [N*m*s/rad], original: 0.0015, identified: 5e-6
            Mp=0.024,  # pendulum link mass [kg]
            Lp=0.129,  # pendulum link length [m]
            Dp=1e-6,  # pendulum link viscous damping [N*m*s/rad], original: 0.0005, identified: 1e-6
            V_thold_neg=0,  # min. voltage required to move the servo in negative the direction [V]
            V_thold_pos=0,  # min. voltage required to move the servo in positive the direction [V]
        )

    def _mujoco_step(self, act: np.ndarray) -> dict:
        assert self.act_space.contains(act, verbose=True)

        # Apply a voltage dead zone, i.e. below a certain amplitude the system is will not move.
        # This is a very simple model of static friction.
        if self.domain_param["V_thold_neg"] <= act <= self.domain_param["V_thold_pos"]:
            act = 0

        km = self.domain_param["km"]
        Rm = self.domain_param["Rm"]

        # Decompose state
        _, _, theta_dot, _ = self.state

        # Compute the torques for the PD controller and clip them to their max values
        torque = (
            km * (float(act) - km * theta_dot) / Rm
        )  # act is a scalar array, causing warning on later np.array construction

        # Apply the torques to the robot
        self.sim.data.ctrl[:] = torque

        # Call MuJoCo
        try:
            self.sim.step()
            mjsim_crashed = False
        except mujoco_py.builder.MujocoException:
            # When MuJoCo recognized instabilities in the simulation, it simply kills it.
            # Instead, we want the episode to end with a failure.
            mjsim_crashed = True

        qpos = self.sim.data.qpos.copy()
        qvel = self.sim.data.qvel.copy()
        self.state = np.concatenate([qpos, qvel])

        # If state is out of bounds (this is normally checked by the task, but does not work because of the mask)
        state_oob = not self.state_space.contains(self.state)

        return dict(
            qpos=qpos,
            qvel=qvel,
            failed=mjsim_crashed or state_oob,
        )

    def observe(self, state: np.ndarray, dtype: type = np.ndarray):
        return np.array([np.sin(state[0]), np.cos(state[0]), np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    def _adapt_model_file(self, xml_model: str, domain_param: dict) -> str:
        xml_model = xml_model.replace("[0.13-Lp]", str(0.13 - domain_param["Lp"]))
        xml_model = xml_model.replace("[0.0055+length_rot_pole]", str(0.0055 + domain_param["length_rot_pole"]))
        return super()._adapt_model_file(xml_model, domain_param)


class QQubeSwingUpMjSim(QQubeMjSim):
    name: str = "qq-mj-su"

    @property
    def state_space(self) -> Space:
        max_state = np.array([115.0 / 180 * np.pi, 4 * np.pi, 20 * np.pi, 20 * np.pi])  # [rad, rad, rad/s, rad/s]
        return BoxSpace(-max_state, max_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])

    @property
    def init_space(self) -> Space:
        max_init_state = np.array([2.0, 1.0, 0.5, 0.5]) / 180 * np.pi  # [rad, rad, rad/s, rad/s]
        return BoxSpace(-max_init_state, max_init_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([1.0, 1.0, 2e-2, 5e-3]))  # former: [3e-1, 1.0, 2e-2, 5e-3]
        R = task_args.get("R", np.diag([4e-3]))

        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])


class QQubeStabMjSim(QQubeMjSim):
    name: str = "qq-mj-st"

    @property
    def state_space(self) -> Space:
        max_state = np.array([115.0 / 180 * np.pi, 4 * np.pi, 20 * np.pi, 20 * np.pi])  # [rad, rad, rad/s, rad/s]
        return BoxSpace(-max_state, max_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])

    @property
    def init_space(self) -> Space:
        min_init_state = np.array([-5.0 / 180 * np.pi, 175.0 / 180 * np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]
        max_init_state = np.array([5.0 / 180 * np.pi, 185.0 / 180 * np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]
        return BoxSpace(min_init_state, max_init_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([3.0, 4.0, 2.0, 2.0]))
        R = task_args.get("R", np.diag([5e-2]))

        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])
