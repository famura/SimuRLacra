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
from typing import Optional

import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.mujoco.base import MujocoSimEnv
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.reward_functions import ForwardVelocityRewFcnHumanoid


class HumanoidSim(MujocoSimEnv, Serializable):
    """
    The Humanoid (v3) MuJoCo simulation environment where a humanoid robot tries to run forward.

    .. seealso::
        [1] https://github.com/openai/gym/blob/master/gym/envs/mujoco/humanoid_v3.py
    """

    name: str = "hum"

    def __init__(
        self,
        frame_skip: int = 5,
        dt: Optional[float] = None,
        max_steps: Optional[int] = 1000,
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
        :param task_args: arguments for the task construction, e.g `dict(fwd_rew_weight=1.)`
        """
        self._exclude_current_positions_from_observation = True

        # Call MujocoSimEnv's constructor
        model_path = osp.join(osp.dirname(__file__), "assets", "openai_humanoid.xml")
        super().__init__(model_path, frame_skip, dt, max_steps, task_args)

        # Initial state
        noise_halfspan = self.domain_param["reset_noise_halfspan"]
        min_init_qpos = self.init_qpos - np.full_like(self.init_qpos, noise_halfspan)
        max_init_qpos = self.init_qpos + np.full_like(self.init_qpos, noise_halfspan)
        min_init_qvel = self.init_qvel - np.full_like(self.init_qvel, noise_halfspan)
        max_init_qvel = self.init_qvel + np.full_like(self.init_qvel, noise_halfspan)

        rest_shape = np.concatenate(
            [
                self.sim.data.cinert.flat,
                self.sim.data.cvel.flat,
                self.sim.data.qfrc_actuator.flat,
                self.sim.data.cfrc_ext.flat,
            ]
        ).shape
        min_init_rest = -np.ones(rest_shape)
        max_init_rest = np.ones(rest_shape)
        min_init_state = np.concatenate([min_init_qpos, min_init_qvel, min_init_rest]).ravel()
        max_init_state = np.concatenate([max_init_qpos, max_init_qvel, max_init_rest]).ravel()
        self._init_space = BoxSpace(min_init_state, max_init_state)

        self.camera_config = dict(distance=5.0)

    @property
    def state_space(self) -> Space:
        state_shape = np.concatenate(
            [
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.sim.data.cinert.flat,
                self.sim.data.cvel.flat,
                self.sim.data.qfrc_actuator.flat,
                self.sim.data.cfrc_ext.flat,
            ]
        ).shape
        return BoxSpace(-pyrado.inf, pyrado.inf, shape=state_shape)

    @property
    def obs_space(self) -> Space:
        obs_shape = self.observe(self.state_space.bound_up).shape
        return BoxSpace(-pyrado.inf, pyrado.inf, shape=obs_shape)

    @property
    def act_space(self) -> Space:
        act_bounds = self.model.actuator_ctrlrange.copy().T
        return BoxSpace(
            *act_bounds,
            labels=[
                "abdomen_y",
                "abdomen_z",
                "abdomen_x",
                "right_hip_x",
                "right_hip_z",
                "right_hip_y",
                "right_knee",
                "left_hip_x",
                "left_hip_z",
                "left_hip_y",
                "left_knee",
                "right_shoulder1",
                "right_shoulder2",
                "right_elbow",
                "left_shoulder1",
                "left_shoulder2",
                "left_elbow",
            ],
        )

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            reset_noise_halfspan=0.0,  # fixed initial state by default
            gravity=9.81,
            sliding_friction=1,
            torsional_friction=0.005,
            rolling_friction=0.0001,
            density=1000,  # scales linearly with the mass
            wind_x=0,
            wind_y=0,
            wind_z=0,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        return GoallessTask(self.spec, ForwardVelocityRewFcnHumanoid(self._dt, **task_args))

    def _mujoco_step(self, act: np.ndarray) -> dict:
        self.sim.data.ctrl[:] = act
        self.sim.step()

        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        com_inertia = self.sim.data.cinert.flat.copy()
        com_velocity = self.sim.data.cvel.flat.copy()
        actuator_forces = self.sim.data.qfrc_actuator.flat.copy()
        external_contact_forces = self.sim.data.cfrc_ext.flat.copy()

        self.state = np.concatenate(
            [
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            ]
        )

        return dict()

    def observe(self, state: np.ndarray) -> np.ndarray:
        position = state[: self.init_qpos.size].copy()
        stateWithoutPos = state[self.init_qpos.size :].copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, stateWithoutPos))
