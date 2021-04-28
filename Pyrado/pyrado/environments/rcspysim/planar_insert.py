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

import functools
import os.path as osp
from typing import Callable, Sequence

import numpy as np
import rcsenv
from init_args_serializer import Serializable

from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.final_reward import FinalRewMode, FinalRewTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.predefined import create_task_space_discrepancy_task
from pyrado.tasks.reward_functions import AbsErrRewFcn, ExpQuadrErrRewFcn, RewFcn
from pyrado.tasks.utils import proximity_succeeded
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "PlanarInsert"))


def create_insert_task(env_spec: EnvSpec, state_des: np.ndarray, rew_fcn: RewFcn, success_fcn: Callable):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ["Effector_X", "Effector_Z", "Effector_B", "Effector_Xd", "Effector_Zd", "Effector_Bd"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space, env_spec.act_space, env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a wrapped desired state task with the goal behind the wall
    fdst = FinalRewTask(
        DesStateTask(spec, state_des, rew_fcn, success_fcn),
        mode=FinalRewMode(state_dependent=True, time_dependent=True),
    )

    # Mask selected states
    return MaskedTask(env_spec, fdst, idcs)


class PlanarInsertSim(RcsSim, Serializable):
    """
    Planar 5- or 6-link robot environment where the task is to push the wedge-shaped end-effector through a small gap
    """

    def __init__(self, task_args: dict, collision_config: dict = None, max_dist_force: float = None, **kwargs):

        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param max_dist_force: maximum disturbance force, pass `None` for no disturbance
        :param kwargs: keyword arguments forwarded to `RcsSim`
                       collisionConfig: specification of the Rcs CollisionModel
        """
        Serializable._init(self, locals())

        # Forward to RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType="PlanarInsert",
            physicsConfigFile="pPlanarInsert.xml",
            collisionConfig=collision_config,
            **kwargs,
        )

        if kwargs.get("collisionConfig", None) is None:
            collision_config = {
                "pairs": [
                    {"body1": "Effector", "body2": "Link3"},
                    {"body1": "Effector", "body2": "Link2"},
                    {"body1": "UpperWall", "body2": "Link4"},
                    {"body1": "LowerWall", "body2": "Link4"},
                    {"body1": "LowerWall", "body2": "Link3"},
                    {"body1": "LowerWall", "body2": "Link2"},
                ],
                "threshold": 0.05,
            }
        else:
            collision_config = kwargs.get("collisionConfig")

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", None)
        if state_des is None:
            # Get and set goal position in world coordinates
            p = self.get_body_position("Goal", "", "")
            state_des = np.array([p[0], p[2], 0, 0, 0, 0])  # X, Z, B, Xd, Zd, Bd

        # Create the individual subtasks
        task_reach_goal = create_insert_task(
            self.spec,
            state_des,
            rew_fcn=ExpQuadrErrRewFcn(
                Q=np.diag([2e1, 2e1, 1e-1, 1e-2, 1e-2, 1e-2]), R=2e-2 * np.eye(self.act_space.flat_dim)
            ),
            success_fcn=functools.partial(proximity_succeeded, thold_dist=0.07, dims=[0, 1, 2]),  # position and angle
        )
        task_ts_discrepancy = create_task_space_discrepancy_task(
            self.spec, AbsErrRewFcn(q=0.1 * np.ones(2), r=np.zeros(self.act_space.shape))
        )
        return ParallelTasks([task_reach_goal, task_ts_discrepancy])

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(
            link1_mass=2.0,
            link2_mass=2.0,
            link3_mass=2.0,
            link4_mass=2.0,
            upperwall_pos_offset_x=0.0,
            upperwall_friction_coefficient=0.5,
            effector_friction_coefficient=0.8,
        )

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        # Sample angle and force uniformly
        angle = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(0, self._max_dist_force)
        return np.array([force * np.sin(angle), 0, force * np.cos(angle)])


class PlanarInsertIKActivationSim(PlanarInsertSim, Serializable):
    """Planar 5- or 6-link robot environment controlled by  setting the input to an Rcs IK-based controller"""

    name: str = "pi-ika"

    def __init__(self, state_des: np.ndarray = None, **kwargs):
        """
        Constructor

        :param state_des: desired state for the task, pass `None` to use the default goal
        :param kwargs: keyword arguments forwarded to `RcsSim`
                       graphFileName: str = 'gPlanarInsert5Link.xml' or 'gPlanarInsert6Link.xml'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeForceTorque: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemGoalDistance: bool = False,
                       observeDynamicalSystemDiscrepancy: bool = False,
        """
        Serializable._init(self, locals())

        dt = kwargs.get("dt", 0.01)  # 100 Hz is the default
        task_spec_ik = [
            dict(x_des=np.array([dt * 0.1])),
            dict(x_des=np.array([dt * 0.1])),
            dict(x_des=np.array([dt * 5 / 180 * np.pi])),
        ]

        # Forward to the PlanarInsertSim's constructor, nothing more needs to be done here
        PlanarInsertSim.__init__(
            self,
            task_args=dict(state_des=state_des),
            actionModelType="ik_activation",
            taskSpecIK=task_spec_ik,
            **kwargs,
        )


class PlanarInsertTASim(PlanarInsertSim, Serializable):
    """Planar 5- or 6-link robot environment controlled by setting the task activation of a Rcs control task"""

    name: str = "pi-ta"

    def __init__(self, mps: Sequence[dict] = None, state_des: np.ndarray = None, **kwargs):
        """
        Constructor

        :param mps: movement primitives for the cartesian x and z velocity and the angular velocity around y-axis
        :param state_des: desired state for the task, pass `None` to use the default goal
        :param kwargs: keyword arguments forwarded to `RcsSim`
                       graphFileName: str = 'gPlanarInsert5Link.xml' or 'gPlanarInsert6Link.xml'
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeForceTorque: bool = True,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemGoalDistance: bool = False,
                       observeDynamicalSystemDiscrepancy: bool = False,
        """
        Serializable._init(self, locals())

        # Define the movement primitives
        dt = kwargs.get("dt", 0.01)  # 100 Hz is the default
        if mps is None:
            mps = [
                # Xd
                {"function": "lin", "errorDynamics": 2.0, "goal": dt * 0.1},  # [m/s]
                {"function": "lin", "errorDynamics": 2.0, "goal": -dt * 0.1},  # [m/s]
                # Zd
                {"function": "lin", "errorDynamics": 2.0, "goal": dt * 0.1},  # [m/s]
                {"function": "lin", "errorDynamics": 2.0, "goal": -dt * 0.1},  # [m/s]
                # Bd
                {"function": "lin", "errorDynamics": 2.0, "goal": dt * 10.0 / 180 * np.pi},  # [rad/s]
                {"function": "lin", "errorDynamics": 2.0, "goal": -dt * 10.0 / 180 * np.pi},  # [rad/s]
            ]

        # Forward to the PlanarInsertSim's constructor, nothing more needs to be done here
        PlanarInsertSim.__init__(
            self, task_args=dict(state_des=state_des), actionModelType="ds_activation", tasks=mps, **kwargs
        )
