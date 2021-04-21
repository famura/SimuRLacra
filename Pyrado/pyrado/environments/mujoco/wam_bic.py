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

import mujoco_py
import numpy as np
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.barrett_wam import (
    act_space_bic_4dof,
    act_space_bic_7dof,
    goal_pos_init_sim_4dof,
    goal_pos_init_sim_7dof,
    init_qpos_des_4dof,
    init_qpos_des_7dof,
    wam_dgains_4dof,
    wam_dgains_7dof,
    wam_pgains_4dof,
    wam_pgains_7dof,
    wam_q_limits_lo_7dof,
    wam_q_limits_up_7dof,
)
from pyrado.environments.mujoco.wam_base import WAMSim
from pyrado.spaces.base import Space
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.tasks.condition_only import ConditionOnlyTask
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.final_reward import BestStateFinalRewTask, FinalRewMode, FinalRewTask
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, QuadrErrRewFcn, ZeroPerStepRewFcn
from pyrado.tasks.sequential import SequentialTasks
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt


class WAMBallInCupSim(WAMSim, Serializable):
    """
    WAM robotic arm from Barrett technologies for the ball-in-the-cup task, controlled by a PD controller.

    .. note::
        When using the `reset()` function, always pass a meaningful `init_state`

    .. seealso::
        [1] https://github.com/psclklnk/self-paced-rl/tree/master/sprl/envs/ball_in_a_cup.py
    """

    name: str = "wam-bic"

    def __init__(
        self,
        num_dof: int,
        frame_skip: int = 4,
        dt: Optional[float] = None,
        max_steps: int = pyrado.inf,
        fixed_init_state: bool = True,
        stop_on_collision: bool = True,
        observe_ball: bool = False,
        observe_cup: bool = False,
        task_args: Optional[dict] = None,
    ):
        """
        Constructor

        :param num_dof: number of degrees of freedom (4 or 7), depending on which Barrett WAM setup being used
        :param frame_skip: number of simulation frames for which the same action is held, results in a multiplier of
                           the time step size `dt`
        :param dt: by default the time step size is the one from the mujoco config file multiplied by the number of
                   frame skips (legacy from OpenAI environments). By passing an explicit `dt` value, this can be
                   overwritten. Possible use case if if you know that you recorded a trajectory with a specific `dt`.
        :param max_steps: max number of simulation time steps
        :param fixed_init_state: enables/disables deterministic, fixed initial state
        :param stop_on_collision: set the `failed` flag in the `dict` returned by `_mujoco_step()` to true, if the ball
                                  collides with something else than the desired parts of the cup. This causes the
                                  episode to end. Keep in mind that in case of a negative step reward and no final
                                  cost on failing, this might result in undesired behavior.
        :param observe_ball: if `True`, include the 2-dim (x-z plane) cartesian ball position into the observation
        :param observe_cup: if `True`, include the 2-dim (x-z plane) cartesian cup position into the observation
        :param task_args: arguments for the task construction
        """
        Serializable._init(self, locals())

        self.fixed_init_state = fixed_init_state
        self.observe_ball = observe_ball
        self.observe_cup = observe_cup

        # Initialize num DoF specific variables
        if num_dof == 4:
            graph_file_name = "wam_4dof_bic.xml"
            self.qpos_des_init = init_qpos_des_4dof
            self.p_gains = wam_pgains_4dof
            self.d_gains = wam_dgains_4dof
            init_ball_pos = np.array([0.723, 0.0, 1.168])
            init_cup_goal = goal_pos_init_sim_4dof
        elif num_dof == 7:
            graph_file_name = "wam_7dof_bic.xml"
            self.qpos_des_init = init_qpos_des_7dof
            self.p_gains = wam_pgains_7dof
            self.d_gains = wam_dgains_7dof
            init_ball_pos = np.array([0.828, 0.0, 1.131])
            init_cup_goal = goal_pos_init_sim_7dof
        else:
            raise pyrado.ValueErr(given=num_dof, eq_constraint="4 or 7")

        model_path = osp.join(pyrado.MUJOCO_ASSETS_DIR, graph_file_name)
        super().__init__(num_dof, model_path, frame_skip, dt, max_steps, task_args)

        # Actual initial joint position (when the WAM moved to the home position)
        if num_dof == 4:
            self.init_qpos[:4] = np.array([0.0, 0.63, 0.0, 1.27])
            self.init_qpos[4] = -0.34  # angle of the first rope segment relative to the cup bottom plate
        else:
            self.init_qpos[:7] = np.array([0.0, 0.65, 0.0, 1.41, 0.0, -0.28, -1.57])
            self.init_qpos[7] = -0.21  # angle of the first rope segment relative to the cup bottom plate

        # Set the actual stable initial position. This position would be reached after some time using the internal
        # PD controller to stabilize at self._qpos_des_init.
        # The initial position of the ball in cartesian coordinates
        self._init_state = np.concatenate([self.init_qpos, self.init_qvel, init_ball_pos, init_cup_goal])
        if self.fixed_init_state:
            self._init_space = SingularStateSpace(self._init_state)
        else:
            # Add plus/minus one degree to each motor joint and the first rope segment joint
            init_state_up = self._init_state.copy()
            init_state_up[: self._num_dof] += np.pi / 180 * np.array([0.1, 1, 0.5, 1.0, 0.1, 1.0, 1.0])[: self._num_dof]
            init_state_lo = self._init_state.copy()
            init_state_lo[: self._num_dof] -= np.pi / 180 * np.array([0.1, 1, 0.5, 1.0, 0.1, 1.0, 1.0])[: self._num_dof]
            self._init_space = BoxSpace(init_state_lo, init_state_up)

        # Bodies to check fo collision
        self._collision_bodies = [
            "wam/base_link",
            "wam/shoulder_yaw_link",
            "wam/shoulder_pitch_link",
            "wam/upper_arm_link",
            "wam/forearm_link",
            "wrist_palm_link",
            "wam/wrist_pitch_link",
            "wam/wrist_yaw_link",
        ]
        if self._num_dof == 4:
            self._collision_bodies = self._collision_bodies[:6]

        # We access a private attribute since a method like 'model.geom_names[geom_id]' cannot be used because
        # not every geom has a name
        self._collision_geom_ids = [self.model._geom_name2id[name] for name in ["cup_geom1", "cup_geom2"]]
        self.stop_on_collision = stop_on_collision

        self.camera_config.update(dict(distance=2.7))

    @property
    def state_space(self) -> Space:
        # The state space has the same shape as the init space (including ball and cup)
        state_shape = np.concatenate([self.init_qpos, self.init_qvel, np.empty(3), np.empty(3)]).shape
        state_lo, state_up = np.full(state_shape, -pyrado.inf), np.full(state_shape, pyrado.inf)

        # Ensure that joint limits of the arm are not reached (5 deg safety margin)
        state_lo[: self._num_dof] = wam_q_limits_lo_7dof[: self._num_dof]
        state_up[: self._num_dof] = wam_q_limits_up_7dof[: self._num_dof]

        return BoxSpace(state_lo, state_up)

    @property
    def obs_space(self) -> Space:
        # Observing the normalized time and optionally the cup and ball position
        obs_lo, obs_up, labels = [0.0], [1.0], ["t"]
        if self.observe_ball:
            obs_lo.extend([-3.0, -3.0])
            obs_up.extend([3.0, 3.0])
            labels.extend(["ball_x", "ball_z"])
        if self.observe_cup:
            obs_lo.extend([-3.0, -3.0])
            obs_up.extend([3.0, 3.0])
            labels.extend(["cup_x", "cup_z"])
        return BoxSpace(obs_lo, obs_up, labels=labels)

    @property
    def act_space(self) -> Space:
        # Running a PD controller on joint positions and velocities
        return act_space_bic_7dof if self._num_dof == 7 else act_space_bic_4dof

    @classmethod
    def get_nominal_domain_param(cls, num_dof: int = 7) -> dict:
        domain_param = super().get_nominal_domain_param()
        domain_param.update(
            dict(
                cup_scale=1.0,  # scaling factor for the radius of the cup [-] (should be >0.65)
                rope_length=0.41,  # length of the rope [m]
                ball_mass=0.024,  # mass of the ball [kg]
                rope_damping=1e-4,  # damping of rope joints [N/s] (reasonable values are 6e-4 to 1e-6)
            )
        )
        return domain_param

    def _create_task(self, task_args: dict) -> Task:
        if task_args.get("sparse_rew_fcn", False):
            # Create a task with binary reward
            return self._create_main_task(task_args)
        else:
            # Create two (or three) parallel running task.
            #   1.) Main task: Desired state task for the cartesian ball distance
            #   2.) Deviation task: Desired state task for the cartesian- and joint deviation from the init position
            #   3.) Binary Bonus: Adds a binary bonus when ball is catched [inactive by default]
            return ParallelTasks(
                [
                    self._create_main_task(task_args),
                    self._create_deviation_task(task_args),
                    self._create_main_task(
                        dict(
                            sparse_rew_fcn=True,
                            success_bonus=task_args.get("success_bonus", 0),
                        )
                    ),
                ]
            )

    def _create_main_task(self, task_args: dict) -> Task:
        # Create a DesStateTask that masks everything but the ball position
        idcs = list(range(self.state_space.flat_dim - 6, self.state_space.flat_dim - 3))  # Cartesian ball position
        spec = EnvSpec(
            self.spec.obs_space,
            self.spec.act_space,
            self.spec.state_space.subspace(self.spec.state_space.create_mask(idcs)),
        )

        # If we do not use copy(), state_des coming from MuJoCo is a reference and updates automatically at each step.
        # Note: sim.forward() + get_body_xpos() results in wrong output for state_des, as sim has not been updated to
        # init_space.sample(), which is first called in reset()

        if task_args.get("sparse_rew_fcn", False):
            factor = task_args.get("success_bonus", 1)
            # Binary final reward task
            main_task = FinalRewTask(
                ConditionOnlyTask(
                    spec,
                    condition_fcn=self.check_ball_in_cup,
                    is_success_condition=True,
                ),
                mode=FinalRewMode(always_positive=True),
                factor=factor,
            )
            # Yield -1 on fail after the main task ist done (successfully or not)
            dont_fail_after_succ_task = FinalRewTask(
                GoallessTask(spec, ZeroPerStepRewFcn()),
                mode=FinalRewMode(always_negative=True),
                factor=factor,
            )

            # Augment the binary task with an endless dummy task, to avoid early stopping
            task = SequentialTasks((main_task, dont_fail_after_succ_task))

            return MaskedTask(self.spec, task, idcs)

        else:
            state_des = self.sim.data.get_site_xpos("cup_goal")  # this is a reference
            # state_des_ball = self.sim.data.get_site_xpos("cup_goal")  # this is a reference
            # state_des_cup = np.array([0.82521, 0, 1.4469]) if self._num_dof == 7 else np.array([0.758, 0, 1.5])
            # state_des = np.concatenate([state_des_ball, state_des_cup])
            R_default = np.diag([0, 0, 1, 1e-2, 1e-2, 1e-1]) if self._num_dof == 7 else np.diag([0, 0, 1e-2, 1e-2])
            rew_fcn = ExpQuadrErrRewFcn(
                Q=task_args.get("Q", np.diag([2e1, 1e-4, 2e1])),  # distance ball - cup; shouldn't move in y-direction
                R=task_args.get("R", R_default),  # last joint is really unreliable for 7 dof, thus punish more
            )
            task = DesStateTask(spec, state_des, rew_fcn)

            # Wrap the masked DesStateTask to add a bonus for the best state in the rollout
            return BestStateFinalRewTask(
                MaskedTask(self.spec, task, idcs),
                factor=task_args.get("final_factor", 0.05 * self.max_steps),
            )

    def _create_deviation_task(self, task_args: dict) -> Task:
        idcs = list(range(self.state_space.flat_dim - 3, self.state_space.flat_dim))  # Cartesian cup goal position
        spec = EnvSpec(
            self.spec.obs_space,
            self.spec.act_space,
            self.spec.state_space.subspace(self.spec.state_space.create_mask(idcs)),
        )
        # init cup goal position
        state_des = goal_pos_init_sim_7dof if self._num_dof == 7 else goal_pos_init_sim_4dof
        rew_fcn = QuadrErrRewFcn(
            Q=task_args.get("Q_dev", np.diag([2e-1, 1e-6, 5e0])),  # Cartesian distance from init cup position
            R=task_args.get(
                "R_dev", np.zeros((self.act_space.shape[0], self.act_space.shape[0]))
            ),  # joint space distance from init pose, interferes with R_default from _create_main_task
        )
        task = DesStateTask(spec, state_des, rew_fcn)

        return MaskedTask(self.spec, task, idcs)

    def _adapt_model_file(self, xml_model: str, domain_param: dict) -> str:
        # First replace special domain parameters
        cup_scale = domain_param.pop("cup_scale", None)
        rope_length = domain_param.pop("rope_length", None)

        if cup_scale is not None:
            # See [1, l.93-96]
            xml_model = xml_model.replace("[scale_mesh]", str(cup_scale * 0.001))
            xml_model = xml_model.replace("[pos_mesh]", str(0.055 - (cup_scale - 1.0) * 0.023))
            xml_model = xml_model.replace("[pos_goal]", str(0.1165 + (cup_scale - 1.0) * 0.0385))
            xml_model = xml_model.replace("[size_cup]", str(cup_scale * 0.038))
            xml_model = xml_model.replace("[size_cup_inner]", str(cup_scale * 0.03))

        if rope_length is not None:
            # The rope consists of 30 capsules
            xml_model = xml_model.replace("[pos_capsule]", str(rope_length / 30))
            # Each joint is at the top of each capsule (therefore negative direction from center)
            xml_model = xml_model.replace("[pos_capsule_joint]", str(-rope_length / 60))
            # Pure visualization component
            xml_model = xml_model.replace("[size_capsule_geom]", str(rope_length / 72))

        # Resolve mesh directory and replace the remaining domain parameters
        return super()._adapt_model_file(xml_model, domain_param)

    def _mujoco_step(self, act: np.ndarray) -> dict:
        assert self.act_space.contains(act, verbose=True)

        # Get the desired positions and velocities for the selected joints
        qpos_des = self.qpos_des_init.copy()  # the desired trajectory is relative to self._qpos_des_init
        qvel_des = np.zeros_like(qpos_des)
        if self._num_dof == 4:
            np.add.at(qpos_des, [1, 3], act[:2])
            np.add.at(qvel_des, [1, 3], act[2:])
        elif self._num_dof == 7:
            np.add.at(qpos_des, [1, 3, 5], act[:3])
            np.add.at(qvel_des, [1, 3, 5], act[3:])

        # Compute the position and velocity errors
        err_pos = qpos_des - self.state[: self._num_dof]
        err_vel = qvel_des - self.state[self.model.nq : self.model.nq + self._num_dof]

        # Compute the torques for the PD controller and clip them to their max values
        torque = self.p_gains * err_pos + self.d_gains * err_vel
        torque = self.torque_space.project_to(torque)

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
        ball_pos = self.sim.data.get_body_xpos("ball").copy()
        cup_goal = self.sim.data.get_site_xpos("cup_goal").copy()
        self.state = np.concatenate([qpos, qvel, ball_pos, cup_goal])

        # If desired, check for collisions of the ball with the robot
        ball_collided = self.check_ball_collisions() if self.stop_on_collision else False

        # If state is out of bounds (this is normally checked by the task, but does not work because of the mask)
        state_oob = False if self.state_space.contains(self.state) else True

        return dict(
            qpos_des=qpos_des,
            qvel_des=qvel_des,
            qpos=qpos[: self._num_dof],
            qvel=qvel[: self._num_dof],
            ball_pos=ball_pos,
            cup_pos=cup_goal,
            failed=mjsim_crashed or ball_collided or state_oob,
        )

    def check_ball_collisions(self, verbose: bool = False) -> bool:
        """
        Check if an undesired collision with the ball occurs.

        :param verbose: print messages on collision
        :return: `True` if the ball collides with something else than the central parts of the cup
        """
        for i in range(self.sim.data.ncon):
            # Get current contact object
            contact = self.sim.data.contact[i]

            # Extract body-id and body-name of both contact geoms
            body1 = self.model.geom_bodyid[contact.geom1]
            body1_name = self.model.body_names[body1]
            body2 = self.model.geom_bodyid[contact.geom2]
            body2_name = self.model.body_names[body2]

            # Evaluate if the ball collides with part of the WAM (collision bodies)
            # or the connection of WAM and cup (geom_ids)
            c1 = body1_name == "ball" and (
                body2_name in self._collision_bodies or contact.geom2 in self._collision_geom_ids
            )
            c2 = body2_name == "ball" and (
                body1_name in self._collision_bodies or contact.geom1 in self._collision_geom_ids
            )
            if c1 or c2:
                if verbose:
                    print_cbt(
                        f"Undesired collision of {body1_name} and {body2_name} detected!",
                        "y",
                    )
                return True

        return False

    def check_ball_in_cup(self, *args, verbose: bool = False):
        """
        Check if the ball is in the cup.

        :param verbose: print messages when ball is in the cup
        :return: `True` if the ball is in the cup
        """
        for i in range(self.sim.data.ncon):
            # Get current contact object
            contact = self.sim.data.contact[i]

            # Extract body-id and body-name of both contact geoms
            body1 = self.model.geom_bodyid[contact.geom1]
            body1_name = self.model.body_names[body1]
            body2 = self.model.geom_bodyid[contact.geom2]
            body2_name = self.model.body_names[body2]

            # Evaluate if the ball collides with part of the WAM (collision bodies)
            # or the connection of WAM and cup (geom_ids)
            cup_inner_id = self.model._geom_name2id["cup_inner"]
            c1 = body1_name == "ball" and contact.geom2 == cup_inner_id
            c2 = body2_name == "ball" and contact.geom1 == cup_inner_id
            if c1 or c2:
                if verbose:
                    print_cbt(f"The ball is in the cup at time step {self.curr_step}.", "y")
                return True

        return False

    def observe(self, state: np.ndarray) -> np.ndarray:
        # TODO: Debug print-outs, should be removed in future...
        # if self._curr_step == 0:
        #     print_cbt(f'cup xpos: {self.sim.data.get_body_xpos("cup").copy()}', 'b')    # center of frame
        #     print_cbt(f'cup xipos: {self.sim.data.get_body_xipos("cup").copy()}', 'b')  # center of mass

        # Observe the normalized time
        obs = [self._curr_step / self.max_steps]

        # Extract the (x, z) cartesian position of cup and ball (the robot operates in the x-z plane).
        # Note: the cup_goal is the mujoco site object marking the goal position for the ball. It is not identical
        # to the coordinate system origin of the rigid body object 'cup'
        if self.observe_ball:
            obs.extend([state[-3], state[-1]])
        if self.observe_cup:
            obs.extend([state[-6], state[-4]])

        return np.array(obs)
