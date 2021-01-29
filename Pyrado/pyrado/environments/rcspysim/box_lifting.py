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
import os.path as osp
from init_args_serializer import Serializable
from typing import Sequence, Optional

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.predefined import (
    create_check_all_boundaries_task,
    create_lifting_task,
    create_forcemin_task,
    create_flipping_task,
    create_home_pos_task,
)
from pyrado.tasks.sequential import SequentialTasks


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "BoxLifting"))


class BoxLiftingSim(RcsSim, Serializable):
    """ Base class for 2-armed humanoid robot lifting a box out of a basket """

    def __init__(self, task_args: dict, ref_frame: str, **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param ref_frame: reference frame for the MPs, e.g. 'world', 'basket', or 'box'
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       fixedInitState: bool = False,
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType="BoxLifting",
            physicsConfigFile="pBoxLifting.xml",
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "BoxLifting"),
            hudColor="BLACK_RUBBER",
            refFrame=ref_frame,
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        # task_box_flip = create_flipping_task(self.spec, ["Box_A"], des_angle_delta=np.pi / 2.0, endless=False)
        task_box_lift = create_lifting_task(self.spec, ["Box_Z"], des_height=0.79, succ_thold=0.05)
        task_post_lift = create_home_pos_task(
            self.spec, ["PowerGrasp_R_Y", "PowerGrasp_R_Z"], state_des=np.array([-0.1, 1.1])
        )
        tasks_box = SequentialTasks([task_box_lift, task_post_lift], hold_rew_when_done=True, verbose=True)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        task_force = create_forcemin_task(
            self.spec, ["WristLoadCellLBR_R_Fy", "WristLoadCellLBR_R_Fz"], Q=np.diag([1e-6, 1e-6])
        )
        # task_collision = create_collision_task(self.spec, factor=1.0)
        # task_ts_discrepancy = create_task_space_discrepancy_task(
        #     self.spec, AbsErrRewFcn(q=0.5 * np.ones(2), r=np.zeros(self.act_space.shape))
        # )

        return ParallelTasks(
            [
                tasks_box,
                task_check_bounds,
                # task_force,
                # task_collision,
                # task_ts_discrepancy
            ],
            hold_rew_when_done=False,
        )

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(
            box_length=0.14,
            box_width=0.18,
            box_mass=0.3,
            box_friction_coefficient=1.4,
            basket_mass=0.5,
            basket_friction_coefficient=0.6,
        )


class BoxLiftingPosIKActivationSim(BoxLiftingSim, Serializable):
    """ Humanoid robot lifting a box out of a basket using a position-level Rcs IK-based controller """

    name: str = "bl-ika-pos"

    def __init__(self, ref_frame: str, **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the Rcs tasks, e.g. 'world', 'table', or 'box'
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       checkJointLimits:m bool = False,
                       collisionAvoidanceIK: bool = True,
                       positionTasks: bool = True,
                       observeVelocities: bool = False,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        task_spec_ik = [
            dict(x_des=np.array([-0.4])),  # Y right
            dict(x_des=np.array([0.3])),  # Z right
            dict(x_des=np.array([0.0])),  # distance Box
        ]

        # Forward to the BoxLiftingSim's constructor
        super().__init__(
            task_args=dict(),
            ref_frame=ref_frame,
            actionModelType="ik_activation",
            taskSpecIK=task_spec_ik,
            positionTasks=True,
            **kwargs,
        )


class BoxLiftingVelIKActivationSim(BoxLiftingSim, Serializable):
    """ Humanoid robot lifting a box out of a basket using a velocity-level Rcs IK-based controller """

    name: str = "bl-ika-vel"

    def __init__(self, ref_frame: str, **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the Rcs tasks, e.g. 'world', 'table', or 'box'
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       checkJointLimits:m bool = False,
                       collisionAvoidanceIK: bool = True,
                       positionTasks: bool = True,
                       observeVelocities: bool = False,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        dt = kwargs.get("dt", 0.01)  # 100 Hz is the default
        task_spec_ik = [
            dict(x_des=np.array([+dt * 0.1])),  # Yd right
            dict(x_des=np.array([+dt * 0.1])),  # Zd right
            dict(x_des=np.array([0.0])),  # X (always active)
            # dict(x_des=np.array([0.0])),  # Cd (always active)
        ]

        # Forward to the BoxLiftingSim's constructor
        super().__init__(
            task_args=dict(),
            ref_frame=ref_frame,
            actionModelType="ik_activation",
            taskSpecIK=task_spec_ik,
            positionTasks=False,
            **kwargs,
        )


class BoxLiftingPosDSSim(BoxLiftingSim, Serializable):
    """ Humanoid robot lifting a box out of a basket using two arms and position-level movement primitives """

    name: str = "bl-ds-pos"

    def __init__(
        self, ref_frame: str, tasks_left: Optional[Sequence[dict]], tasks_right: Optional[Sequence[dict]], **kwargs
    ):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'basket', or 'box'
        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param tasks_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       fixedInitState: bool = False,
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        # basket_extends = self.get_body_extents('Basket', 0)
        if tasks_left is None:
            tasks_left = [
                # Power grasp position in basket frame (basket width = 0.7)
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([0.0, 0.5, 0.15]),
                },  # [m]
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([0.0, -0.3, 0.15]),
                },  # [m]
                # Power grasp position in box frame (box width = 0.18)
                # {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                #  'goal': np.array([0., 0., 0.1])},  # [m]
                # Power grasp orientation in basket frame
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.pi / 180 * np.array([180, -90, 0.0]),
                },  # [rad]
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.pi / 180 * np.array([120, -90, 0.0]),
                },  # [rad]
                # Joints SDH
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 50.0,
                    "mass": 1.0,
                    "damping": 50.0,
                    "goal": 10 / 180 * np.pi * np.array([0, 2, -1.5, 2, 0, 2, 0]),
                },
            ]
        if tasks_right is None:
            tasks_right = [
                # Power grasp position in basket frame (basket width = 0.7)
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([0.0, -0.5, 0.15]),
                },  # [m]
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([0.0, 0.3, 0.15]),
                },  # [m]
                # Power grasp orientation
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.pi / 180 * np.array([180, -90, 0.0]),
                },  # [rad]
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.pi / 180 * np.array([240, -90, 0.0]),
                },  # [rad]
                # Joints SDH
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 50.0,
                    "mass": 1.0,
                    "damping": 50.0,
                    "goal": 10 / 180 * np.pi * np.array([0, 1.5, -1, 1, 0, 1.5, 0]),
                },
                # Distance
                # {'function': 'msd', 'attractorStiffness': 50., 'mass': 1., 'damping': 10.,
            ]

        # Forward to the BoxLiftingSim's constructor
        super().__init__(
            task_args=dict(),
            ref_frame=ref_frame,
            actionModelType="ds_activation",
            tasksLeft=tasks_left,
            tasksRight=tasks_right,
            positionTasks=True,
            **kwargs,
        )


class BoxLiftingVelDSSim(BoxLiftingSim, Serializable):
    """ Humanoid robot lifting a box out of a basket using two arms and velocity-level movement primitives """

    name: str = "bl-ds-vel"

    def __init__(
        self, ref_frame: str, tasks_left: Optional[Sequence[dict]], tasks_right: Optional[Sequence[dict]], **kwargs
    ):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'basket', or 'box'
        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param tasks_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       fixedInitState: bool = False,
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeCollisionCost: bool = True,
                       observeVelocities: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemDiscrepancy: bool = False,
                       observeTaskSpaceDiscrepancy: bool = True,
                       observeForceTorque: bool = True
        """
        Serializable._init(self, locals())

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        dt = kwargs.get("dt", 0.01)  # 100 Hz is the default
        # basket_extends = self.get_body_extents('Basket', 0)
        if tasks_left is None:
            tasks_left = [
                # Power grasp Xd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Power grasp Yd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Power grasp Zd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Power grasp Ad
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Power grasp Bd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Power grasp Cd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Joints SDH
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 50.0,
                    "mass": 2.0,
                    "damping": 50.0,
                    "goal": 10 / 180 * np.pi * np.array([0, 2, -1.5, 2, 0, 2, 0]),
                },
            ]
        if tasks_right is None:
            tasks_right = [
                # Power grasp Xd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Power grasp Yd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Power grasp Zd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Power grasp Ad
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Power grasp Bd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Power grasp Cd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Joints SDH
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 50.0,
                    "mass": 2.0,
                    "damping": 50.0,
                    "goal": 10 / 180 * np.pi * np.array([0, 1.5, -1, 1, 0, 1.5, 0]),
                },
            ]

        # Forward to the BoxLiftingSim's constructor
        super().__init__(
            task_args=dict(),
            ref_frame=ref_frame,
            actionModelType="ds_activation",
            tasksLeft=tasks_left,
            tasksRight=tasks_right,
            positionTasks=False,
            **kwargs,
        )
