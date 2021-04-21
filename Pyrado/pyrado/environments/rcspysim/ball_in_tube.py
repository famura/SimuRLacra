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
from typing import Sequence

import numpy as np
import rcsenv
from init_args_serializer import Serializable

from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.predefined import (
    create_check_all_boundaries_task,
    create_collision_task,
    create_task_space_discrepancy_task,
)
from pyrado.tasks.reward_functions import AbsErrRewFcn, ExpQuadrErrRewFcn
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "BallInTube"))


def create_extract_ball_task(env_spec: EnvSpec, task_args: dict, des_state: np.ndarray):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ["Ball_X", "Ball_Y"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space, env_spec.act_space, env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    Q = task_args.get("Q_ball", np.diag([1e1, 1e1]))
    R = task_args.get("R_ball", 1e-6 * np.eye(spec.act_space.flat_dim))
    rew_fcn = ExpQuadrErrRewFcn(Q, R)
    dst_task = DesStateTask(spec, des_state, rew_fcn)

    # Return the masked tasks
    return MaskedTask(env_spec, dst_task, idcs)


def create_extract_slider_task(env_spec: EnvSpec, task_args: dict, des_state: np.ndarray):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ["Slider_Y"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space, env_spec.act_space, env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    Q = task_args.get("Q_slider", np.array([[4e1]]))
    R = task_args.get("R_slider", 1e-6 * np.eye(spec.act_space.flat_dim))
    rew_fcn = ExpQuadrErrRewFcn(Q, R)
    dst_task = DesStateTask(spec, des_state, rew_fcn)

    # Return the masked tasks
    return MaskedTask(env_spec, dst_task, idcs)


class BallInTubeSim(RcsSim, Serializable):
    """ Base class for 2-armed humanoid robot fiddling a ball out of a tube """

    def __init__(self, task_args: dict, ref_frame: str, **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
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

        # collision_config = {
        #     'pairs': [
        #         {'body1': 'Hook_L', 'body2': 'Table'},
        #         {'body1': 'Hook_L', 'body2': 'Slider'},
        #         {'body1': 'Hook_L', 'body2': 'Hook_R'},
        #     ],
        #     'threshold': 0.15,
        #     'predCollHorizon': 20
        # }

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType="BallInTube",
            physicsConfigFile="pBallInTube.xml",
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "BallInTube"),
            collisionConfig={"file": "collisionModel.xml"},
            hudColor="BLACK_RUBBER",
            refFrame=ref_frame,
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        des_state1 = self.get_body_position("Goal", "", "")[:2]  # x and y coordinates in world frame
        task_box = create_extract_ball_task(self.spec, task_args, des_state1)
        des_state2 = np.array(self.get_body_position("Slider", "", "")[1] - 0.5)  # y coordinate in world frame
        task_slider = create_extract_slider_task(self.spec, task_args, des_state2)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        task_collision = create_collision_task(self.spec, factor=5e-2)
        task_ts_discrepancy = create_task_space_discrepancy_task(
            self.spec, AbsErrRewFcn(q=0.5 * np.ones(6), r=np.zeros(self.act_space.flat_dim))
        )

        return ParallelTasks(
            [task_box, task_slider, task_check_bounds, task_collision, task_ts_discrepancy], hold_rew_when_done=False
        )

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(
            ball_mass=0.3,
            ball_radius=0.03,
            ball_rolling_friction_coefficient=0.02,
            # table_friction_coefficient=0.6,
            slider_mass=0.3,
        )


class BallInTubeIKSim(BallInTubeSim, Serializable):
    """ Humanoid robot fiddling a ball out of a tube using two hooks and an position-level IK controller """

    name: str = "bit-ik"

    def __init__(self, ref_frame: str, continuous_rew_fcn: bool = True, **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
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

        # Forward to the BallInTubeSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            actionModelType="ik",
            positionTasks=False,
            **kwargs,
        )


class BallInTubePosIKActivationSim(BallInTubeSim, Serializable):
    """ Humanoid robot fiddling a ball out of a tube using two hooks and an position-level IK controller """

    name: str = "bit-ika-pos"

    def __init__(self, ref_frame: str, continuous_rew_fcn: bool = True, **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
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

        task_spec_ik = [
            dict(x_des=np.array([1.121, -0.0235, 1.2155])),  # left - home X Y Z
            dict(x_des=np.array([0.08, -0.05, 0.10])),  # left - ball X Y Z
            # dict(x_des=np.array([-1.])),  # left - force on ball X
            dict(x_des=np.array([40, 90, -130]) / 180 * np.pi),  # left - orientation A B C
            dict(x_des=np.array([0.902, -0.3821, 1.1949])),  # right - home X Y Z
            dict(x_des=np.array([0.05, 0.02, 0.0])),  # right - slider X Y Z
            dict(x_des=np.array([-0.8])),  # right - slider Y
            dict(x_des=np.array([40, 90, -130]) / 180 * np.pi),  # right - orientation A B C
        ]

        # Forward to the BallInTubeSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            actionModelType="ik_activation",
            positionTasks=True,
            taskSpecIK=task_spec_ik,
            **kwargs,
        )


class BallInTubePosDSSim(BallInTubeSim, Serializable):
    """ Humanoid robot fiddling a ball out of a tube using two hooks and position-level movement primitives """

    name: str = "bit-ds-pos"

    def __init__(
        self,
        ref_frame: str,
        tasks_left: [Sequence[dict], None],
        tasks_right: [Sequence[dict], None],
        continuous_rew_fcn: bool = True,
        **kwargs,
    ):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param tasks_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
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
        if tasks_left is None:
            tasks_left = [
                # Effector position relative to slider
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([-0.05, 0.05, 0.05]),
                },  # [m]
                # Effector orientation relative to slider
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([0.0, 0.0, 0.0]),
                },  # [rad]
            ]
        if tasks_right is None:
            tasks_right = [
                # Effector position relative to slider
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([0.05, -0.05, 0.05]),
                },  # [m]
                # Effector orientation relative to slider
                {
                    "function": "msd_nlin",
                    "attractorStiffness": 30.0,
                    "mass": 1.0,
                    "damping": 60.0,
                    "goal": np.array([0.0, 0.0, 0.0]),
                },  # [rad]
            ]

        # Forward to the BallInTubeSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            actionModelType="ds_activation",
            positionTasks=True,
            tasksLeft=tasks_left,
            tasksRight=tasks_right,
            **kwargs,
        )


class BallInTubeVelDSSim(BallInTubeSim, Serializable):
    """ Humanoid robot fiddling a ball out of a tube using two hooks and velocity-level movement primitives """

    name: str = "bit-ds-vel"

    def __init__(
        self,
        ref_frame: str,
        tasks_left: [Sequence[dict], None],
        tasks_right: [Sequence[dict], None],
        continuous_rew_fcn: bool = True,
        **kwargs,
    ):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param tasks_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
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
        if tasks_left is None:
            tasks_left = [
                # Effector Xd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Effector Yd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Effector Zd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Effector Ad
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Effector Bd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Effector Cd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
            ]
        if tasks_right is None:
            tasks_right = [
                # Effector Xd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Effector Yd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Effector Zd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([0.15])},  # [m/s]
                # Effector Ad
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Effector Bd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
                # Effector Cd
                {"function": "lin", "errorDynamics": 1.0, "goal": dt * np.array([15 / 180 * np.pi])},  # [rad/s]
            ]

        # Forward to the BallInTubeSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            actionModelType="ds_activation",
            positionTasks=False,
            tasksLeft=tasks_left,
            tasksRight=tasks_right,
            **kwargs,
        )
