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
from functools import partial
from typing import Optional

import numpy as np
import rcsenv
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.predefined import create_check_all_boundaries_task
from pyrado.tasks.reward_functions import AbsErrRewFcn
from pyrado.tasks.utils import proximity_succeeded
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "MiniGolf"))


def create_mini_golf_task(env_spec: EnvSpec, hole_pos: np.ndarray, succ_thold: float):
    """
    Create a task for putting the ball into a whole.

    .. note::
        This task was designed with an RcsPySim environment in mind, but is not restricted to these environments.

    :param env_spec: environment specification
    :param hole_pos: planar x and yy  position of the goal's center
    :param succ_thold: once the object of interest is closer than this threshold, the task is considered successfully
    :return: masked task that only considers a subspace of all observations
    """
    if not hole_pos.size == 2:
        raise pyrado.ShapeErr(given=hole_pos, expected_match=(2,))

    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ["Ball_X", "Ball_Y"]

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space, env_spec.act_space, env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    dst = DesStateTask(
        spec,
        state_des=hole_pos,
        rew_fcn=AbsErrRewFcn(q=np.ones(2), r=np.zeros(spec.act_space.shape)),
        success_fcn=partial(proximity_succeeded, thold_dist=succ_thold),
    )

    # Return the masked tasks
    return MaskedTask(env_spec, dst, idcs)


class MiniGolfSim(RcsSim, Serializable):
    """A 7-dof Schunk robot playing mini golf"""

    def __init__(self, task_args: dict, ref_frame: str, **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param ref_frame: reference frame for the Rcs tasks, e.g. 'world' or 'ball'
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       fixedInitState: bool = True,
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeForceTorque: bool = True,
                       observeCollisionCost: bool = False,
                       observePredictedCollisionCost: bool = False,
        """
        Serializable._init(self, locals())

        if kwargs.get("collisionConfig", None) is None:
            collision_config = {
                "pairs": [
                    {"body1": "Club", "body2": "Ground"},
                ],
                "threshold": 0.05,
                "predCollHorizon": 20,
            }
        else:
            collision_config = kwargs.get("collisionConfig")

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            envType="MiniGolf",
            task_args=task_args,
            refFrame=ref_frame,
            graphFileName="gMiniGolf.xml",
            physicsConfigFile="pMiniGolf.xml",
            collisionConfig=collision_config,
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "MiniGolf"),
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        hole_pos = task_args.get("hole_pos", None)
        if hole_pos is None:
            # Get the goal position in world coordinates
            hole_pos = self.get_body_position("Hole", "", "")[:2]  # x and y but not z
        task_main = create_mini_golf_task(self.spec, hole_pos, succ_thold=0.02)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)

        return ParallelTasks([task_main, task_check_bounds], hold_rew_when_done=False)

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(
            ball_mass=0.05,
            ball_friction_coefficient=0.111,
            ball_rolling_friction_coefficient=0.0,
            ground_friction_coefficient=0.777,
        )


class MiniGolfIKSim(MiniGolfSim, Serializable):
    """A 7-dof Schunk robot playing mini golf by setting the input to an Rcs IK-based controller"""

    name: str = "mg-ik"

    def __init__(self, ref_frame: str = "world", task_args: Optional[dict] = None, **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the Rcs tasks, e.g. 'world' or 'ball'
        :param task_args: arguments for the task construction
        :param kwargs: keyword arguments forwarded to `RcsSim`
                       fixedInitState: bool = True,
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeForceTorque: bool = True,
                       observeCollisionCost: bool = False,
                       observePredictedCollisionCost: bool = False,
        """
        Serializable._init(self, locals())

        # Forward to the MiniGolfSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict() if task_args is None else task_args,
            ref_frame=ref_frame,
            actionModelType="ik",
            **kwargs,
        )
