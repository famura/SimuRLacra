from typing import Sequence

import numpy as np
import os.path as osp
from init_args_serializer import Serializable

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

    name: str = 'tt'

    def __init__(self,
                 mps_left: Sequence[dict],
                 mps_right: Sequence[dict],
                 continuous_rew_fcn: bool = True,
                 **kwargs):
        """
        Constructor

        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
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
            envType='TargetTracking',
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn, mps_left=mps_left, mps_right=mps_right),
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'TargetTracking'),
            tasksLeft=mps_left,
            tasksRight=mps_right,
            **kwargs
        )

    def _create_task(self, task_args: dict) -> Task:
        # Set up task. We track the distance to the goal for both hands separately.
        continuous_rew_fcn = task_args.get('continuous_rew_fcn', True)
        mps_left = task_args.get('mps_left')
        mps_right = task_args.get('mps_right')

        if continuous_rew_fcn:
            Q = np.diag([1, 1e-3])
            R = 1e-4*np.eye(self.act_space.flat_dim)
            rew_fcn_factory = lambda: ExpQuadrErrRewFcn(Q, R)
        else:
            rew_fcn_factory = MinusOnePerStepRewFcn
        succ_thold = 7.5e-2

        tasks_left = [
            create_goal_dist_distvel_task(self.spec, i, rew_fcn_factory(), succ_thold)
            for i in range(len(mps_left))
        ]
        tasks_right = [
            create_goal_dist_distvel_task(self.spec, i + len(mps_left), rew_fcn_factory(), succ_thold)
            for i in range(len(mps_right))
        ]

        return ParallelTasks([
            SequentialTasks(tasks_left, hold_rew_when_done=continuous_rew_fcn),
            SequentialTasks(tasks_right, hold_rew_when_done=continuous_rew_fcn),
        ], hold_rew_when_done=continuous_rew_fcn)

    @classmethod
    def get_nominal_domain_param(cls):
        # So far this class is oly used for testing on the real robot
        return {}
