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

import mujoco_py
import numpy as np
import os.path as osp
from init_args_serializer import Serializable
from typing import Optional

import pyrado
from pyrado.environments.barrett_wam import (
    init_qpos_des_7dof,
    init_qpos_des_4dof,
    torque_space_wam_7dof,
    torque_space_wam_4dof,
    wam_pgains_7dof,
    wam_dgains_7dof,
    wam_pgains_4dof,
    wam_dgains_4dof,
    act_space_jsc_7dof,
    act_space_jsc_4dof,
)
from pyrado.environments.mujoco.base import MujocoSimEnv
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn


class WAMJointSpaceCtrlSim(MujocoSimEnv, Serializable):
    """
    WAM robotic arm from Barrett technologies, controlled by a PD controller.

    .. note::
        When using the `reset()` function, always pass a meaningful `init_state`

    .. seealso::
        https://github.com/jhu-lcsr/barrett_model
        http://www.mujoco.org/book/XMLreference.html (e.g. for joint damping)
    """

    name: str = "wam-jsc"

    def __init__(
        self,
        num_dof: int,
        frame_skip: int,
        max_steps: Optional[int] = pyrado.inf,
        task_args: Optional[dict] = None,
    ):
        """
        Constructor

        :param num_dof: number of degrees of freedom (4 or 7), depending on which Barrett WAM setup being used
        :param frame_skip: number of frames for holding the same action, i.e. multiplier of the time step size
        :param max_steps: max number of simulation time steps
        :param task_args: arguments for the task construction
        """
        Serializable._init(self, locals())

        if num_dof == 4:
            graph_file_name = "wam_4dof_base.xml"
            self.qpos_des_init = init_qpos_des_4dof
            self.p_gains = wam_pgains_4dof
            self.d_gains = wam_dgains_4dof
        elif num_dof == 7:
            graph_file_name = "wam_7dof_base.xml"
            self.qpos_des_init = init_qpos_des_7dof
            self.p_gains = wam_pgains_7dof
            self.d_gains = wam_dgains_7dof
        else:
            raise pyrado.ValueErr(given=num_dof, eq_constraint="4 or 7")
        self._num_dof = num_dof

        model_path = osp.join(pyrado.MUJOCO_ASSETS_DIR, graph_file_name)
        super().__init__(model_path, frame_skip, max_steps, task_args)

        self.camera_config = dict(
            trackbodyid=0,  # id of the body to track
            elevation=-30,  # camera rotation around the axis in the plane
            azimuth=-90,  # camera rotation around the camera's vertical axis
        )

    @property
    def num_dof(self) -> int:
        """ Get the number of degrees of freedom. """
        return self._num_dof

    @property
    def torque_space(self) -> Space:
        """ Get the space of joint torques. """
        return self._torque_space

    @classmethod
    def get_nominal_domain_param(cls, num_dof: int = 7) -> dict:
        if num_dof == 7:
            return dict(
                joint_1_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_2_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_3_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_4_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_5_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_6_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_7_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_1_stiction=0.6,  # dry friction coefficient of motor joint 1 [-]
                joint_2_stiction=0.6,  # dry friction coefficient of motor joint 2 [-]
                joint_3_stiction=0.4,  # dry friction coefficient of motor joint 3 [-]
                joint_4_stiction=0.4,  # dry friction coefficient of motor joint 4 [-]
                joint_5_stiction=0.2,  # dry friction coefficient of motor joint 5 [-]
                joint_6_stiction=0.2,  # dry friction coefficient of motor joint 6 [-]
                joint_7_stiction=0.2,  # dry friction coefficient of motor joint 7 [-]
                rope_damping=1e-4,  # damping of rope joints [N/s] (reasonable values are 6e-4 to 1e-6)
            )
        elif num_dof == 4:
            return dict(
                joint_1_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_2_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_3_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_4_damping=0.05,  # damping of motor joints [N/s] (default value is small)
                joint_1_stiction=0.6,  # dry friction coefficient of motor joint 1 [-]
                joint_2_stiction=0.6,  # dry friction coefficient of motor joint 2 [-]
                joint_3_stiction=0.4,  # dry friction coefficient of motor joint 3 [-]
                joint_4_stiction=0.4,  # dry friction coefficient of motor joint 4 [-]
            )
        else:
            raise pyrado.ValueErr(given=num_dof, eq_constraint="4 or 7")

    def _create_spaces(self):
        # Torque and action space (running a PD controller on joint positions and velocities)
        self._torque_space = torque_space_wam_7dof if self._num_dof == 7 else torque_space_wam_4dof
        self._act_space = act_space_jsc_7dof if self._num_dof == 7 else act_space_jsc_4dof

        # State space
        state_shape = np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).shape
        self._state_space = BoxSpace(bound_lo=-pyrado.inf, bound_up=pyrado.inf, shape=state_shape)

        # Fixed initial state space
        init_state = np.concatenate([self.init_qpos, self.init_qvel])
        self._init_space = SingularStateSpace(init_state)

        # Observation space (observations are the exact state)
        self._obs_space = self._state_space.copy()

    def _create_task(self, task_args: Optional[dict] = None) -> Task:
        state_des = np.concatenate([self.init_qpos, self.init_qvel])
        return DesStateTask(self.spec, state_des, ZeroPerStepRewFcn())

    def _mujoco_step(self, act: np.ndarray) -> dict:
        if not act.shape[0] % 2 == 0:
            raise pyrado.ShapeErr(msg=f"The action must have and even length, but its shape is {act.shape}!")

        # Get the desired positions and velocities for the selected joints
        qpos_des, qvel_des = act[: act.shape[0] // 2], act[act.shape[0] // 2 :]

        # Compute the position and velocity errors
        err_pos = qpos_des - self.state[: self._num_dof]
        err_vel = qvel_des - self.state[self.model.nq : self.model.nq + self._num_dof]

        # Compute the torques for the PD controller and clip them to their max values
        torque = self.p_gains * err_pos + self.d_gains * err_vel
        torque = self._torque_space.project_to(torque)

        # Apply the torques to the robot
        self.sim.data.qfrc_applied[: self._num_dof] = torque

        # Call MuJoCo
        try:
            self.sim.step()
            mjsim_crashed = False
        except mujoco_py.builder.MujocoException:
            # When MuJoCo recognized instabilities in the simulation, it simply kills it.
            # Instead, we want the episode to end with a failure.
            mjsim_crashed = True

        qpos, qvel = self.sim.data.qpos.copy(), self.sim.data.qvel.copy()
        self.state = np.concatenate([qpos, qvel])

        # If state is out of bounds (this is normally checked by the task, but does not work because of the mask)
        state_oob = False if self.state_space.contains(self.state) else True

        return dict(
            qpos_des=qpos_des,
            qvel_des=qvel_des,
            qpos=qpos[: self._num_dof],
            qvel=qvel[: self._num_dof],
            failed=mjsim_crashed or state_oob,
        )
