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
from typing import Sequence

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, MinusOnePerStepRewFcn
from pyrado.tasks.sequential import SequentialTasks
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.predefined import create_goal_dist_distvel_task


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)


class TargetTrackingSim(RcsSim, Serializable):
    """ 2-armed humanoid robot going to a target position with both hands """

    name: str = "tt"

    def __init__(
        self, tasks_left: Sequence[dict], tasks_right: Sequence[dict], continuous_rew_fcn: bool = True, **kwargs
    ):
        """
        Constructor

        :param tasks_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param tasks_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeCollisionCost: bool = True,
                       observePredictedCollisionCost: bool = False,
        """
        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            envType="TargetTracking",
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn, tasks_left=tasks_left, tasks_right=tasks_right),
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "TargetTracking"),
            tasksLeft=tasks_left,
            tasksRight=tasks_right,
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Set up task. We track the distance to the goal for both hands separately.
        continuous_rew_fcn = task_args.get("continuous_rew_fcn", True)
        tasks_left = task_args.get("tasks_left")
        tasks_right = task_args.get("tasks_right")

        if continuous_rew_fcn:
            Q = np.diag([1, 1e-3])
            R = 1e-4 * np.eye(self.act_space.flat_dim)
            rew_fcn_factory = lambda: ExpQuadrErrRewFcn(Q, R)
        else:
            rew_fcn_factory = MinusOnePerStepRewFcn
        succ_thold = 7.5e-2

        tasks_left = [
            create_goal_dist_distvel_task(self.spec, i, rew_fcn_factory(), succ_thold) for i in range(len(tasks_left))
        ]
        tasks_right = [
            create_goal_dist_distvel_task(self.spec, i + len(tasks_left), rew_fcn_factory(), succ_thold)
            for i in range(len(tasks_right))
        ]

        return ParallelTasks(
            [
                SequentialTasks(tasks_left, hold_rew_when_done=continuous_rew_fcn),
                SequentialTasks(tasks_right, hold_rew_when_done=continuous_rew_fcn),
            ],
            hold_rew_when_done=continuous_rew_fcn,
        )

    @classmethod
    def get_nominal_domain_param(cls):
        # So far this class is oly used for testing on the real robot
        return {}
