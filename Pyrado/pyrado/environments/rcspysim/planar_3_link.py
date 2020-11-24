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
import numpy as np
import os.path as osp
from init_args_serializer import Serializable
from typing import Sequence

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.predefined import create_check_all_boundaries_task, create_task_space_discrepancy_task
from pyrado.tasks.utils import proximity_succeeded
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, ZeroPerStepRewFcn, AbsErrRewFcn
from pyrado.tasks.sequential import SequentialTasks
from pyrado.tasks.parallel import ParallelTasks
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'Planar3Link'))


class Planar3LinkSim(RcsSim, Serializable):
    """ Base class for the Planar 3-link environments simulated in Rcs using the Vortex or Bullet physics engine """

    def __init__(self, task_args: dict, max_dist_force: float = None, **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param max_dist_force: maximum disturbance force, pass `None` for no disturbance
        :param kwargs: keyword arguments forwarded to `RcsSim`
                       fixedInitState: bool = True,
                       collisionConfig: specification of the Rcs CollisionModel
        """
        Serializable._init(self, locals())

        if kwargs.get('collisionConfig', None) is None:
            collision_config = {
                'pairs': [
                    {'body1': 'Effector', 'body2': 'Link2'},
                    {'body1': 'Effector', 'body2': 'Link1'},
                ],
                'threshold': 0.15,
                'predCollHorizon': 20
            }
        else:
            collision_config = kwargs.get('collisionConfig')

        # Forward to RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType='Planar3Link',
            graphFileName=kwargs.pop('graphFileName', 'gPlanar3Link_trqCtrl.xml'),  # by default torque control
            positionTasks=kwargs.pop('positionTasks', None),  # invalid default value, positionTasks can be unnecessary
            collisionConfig=collision_config,
            **kwargs
        )

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        # Sample angle and force uniformly
        angle = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(0, self._max_dist_force)
        return np.array([force*np.sin(angle), 0, force*np.cos(angle)])

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(
            link1_mass=2.,
            link2_mass=2.,
            link3_mass=2.,
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the indices for selection. This needs to match the observations' names in RcsPySim.
        if task_args.get('consider_velocities', False):
            idcs = ['Effector_X', 'Effector_Z', 'Effector_Xd', 'Effector_Zd']
        else:
            idcs = ['Effector_X', 'Effector_Z']

        # Get the masked environment specification
        spec = EnvSpec(
            self.spec.obs_space,
            self.spec.act_space,
            self.spec.state_space.subspace(self.spec.state_space.create_mask(idcs))
        )

        # Get and set goal position in world coordinates for all three sub-goals
        p1 = self.get_body_position('Goal1', '', '')
        p2 = self.get_body_position('Goal2', '', '')
        p3 = self.get_body_position('Goal3', '', '')
        state_des1 = np.array([p1[0], p1[2], 0, 0])
        state_des2 = np.array([p2[0], p2[2], 0, 0])
        state_des3 = np.array([p3[0], p3[2], 0, 0])
        if task_args.get('consider_velocities', False):
            Q = np.diag([5e-1, 5e-1, 2e-1, 2e-1])
        else:
            Q = np.diag([1e0, 1e0])
            state_des1 = state_des1[:2]
            state_des2 = state_des2[:2]
            state_des3 = state_des3[:2]

        success_fcn = functools.partial(proximity_succeeded, thold_dist=7.5e-2, dims=[0, 1])  # min distance = 7cm
        R = np.zeros((spec.act_space.flat_dim, spec.act_space.flat_dim))

        # Create the tasks
        subtask_1 = FinalRewTask(
            DesStateTask(spec, state_des1, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_2 = FinalRewTask(
            DesStateTask(spec, state_des2, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_3 = FinalRewTask(
            DesStateTask(spec, state_des3, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_4 = FinalRewTask(
            DesStateTask(spec, state_des1, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_5 = FinalRewTask(
            DesStateTask(spec, state_des2, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        subtask_6 = FinalRewTask(
            DesStateTask(spec, state_des3, ExpQuadrErrRewFcn(Q, R), success_fcn),
            mode=FinalRewMode(time_dependent=True)
        )
        task = FinalRewTask(
            SequentialTasks([subtask_1, subtask_2, subtask_3, subtask_4, subtask_5, subtask_6], hold_rew_when_done=True,
                            verbose=True),
            mode=FinalRewMode(always_positive=True), factor=5e3,
        )
        masked_task = MaskedTask(self.spec, task, idcs)

        # Additional tasks
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        if isinstance(self, Planar3LinkJointCtrlSim):
            # Return the masked task and and additional task that ends the episode if the unmasked state is out of bound
            return ParallelTasks([masked_task, task_check_bounds], easily_satisfied=True)
        else:
            task_ts_discrepancy = create_task_space_discrepancy_task(
                self.spec, AbsErrRewFcn(q=0.5*np.ones(2), r=np.zeros(self.act_space.shape))
            )
            # Return the masked task and and additional task that ends the episode if the unmasked state is out of bound
            return ParallelTasks([masked_task, task_check_bounds, task_ts_discrepancy], easily_satisfied=True)


class Planar3LinkJointCtrlSim(Planar3LinkSim, Serializable):
    """ Planar 3-link robot controlled by directly setting the joint angles """

    name: str = 'p3l-jt'

    def __init__(self, task_args: dict = None, **kwargs):
        """
        Constructor

        :param kwargs: keyword arguments forwarded to `RcsSim`
        """
        Serializable._init(self, locals())

        # Forward to the Planar3LinkSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict() if task_args is None else task_args,
            actionModelType='joint_pos',
            graphFileName='gPlanar3Link_posCtrl.xml',
            **kwargs
        )


class Planar3LinkIKActivationSim(Planar3LinkSim, Serializable):
    """ Planar 3-link robot environment controlled by setting the input to an Rcs IK-based controller """

    name: str = 'p3l-ika'

    def __init__(self, task_args: dict = None, **kwargs):
        """
        Constructor

        :param kwargs: keyword arguments forwarded to `RcsSim`
                       positionTasks: bool = True,
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeForceTorque: bool = True,
                       observeCollisionCost: bool = False,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemGoalDistance: bool = False,
        """
        Serializable._init(self, locals())

        task_spec_ik = [
            dict(x_des=np.array([0., 0., 0.])),
            dict(x_des=np.array([0., 0., 0.])),
            dict(x_des=np.array([0., 0., 0.]))
        ]

        # Forward to the Planar3LinkSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict() if task_args is None else task_args,
            actionModelType='ik_activation',
            taskSpecIK=task_spec_ik,
            **kwargs
        )


class Planar3LinkTASim(Planar3LinkSim, Serializable):
    """ Planar 3-link robot controlled by setting the task activation of a Rcs control task """

    name: str = 'p3l-ta'

    def __init__(self,
                 task_args: dict = None,
                 mps: Sequence[dict] = None,
                 position_mps: bool = True,
                 **kwargs):
        """
        Constructor

        :param mps: movement primitives holding the dynamical systems and the goal states
        :param position_mps: if `True` use movement primitives specified on position-level, if `False` velocity-level
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'sum',  # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
                       observeVelocities: bool = True,
                       observeForceTorque: bool = True,
                       observeCollisionCost: bool = False,
                       observePredictedCollisionCost: bool = False,
                       observeManipulabilityIndex: bool = False,
                       observeCurrentManipulability: bool = True,
                       observeDynamicalSystemGoalDistance: bool = False,
                       observeDynamicalSystemDiscrepancy: bool = False,

        Example:
        mps = [{'function': 'lin',
                'errorDynamics': np.eye(dim_mp_state),
                'goal': np.zeros(dim_mp_state)
               },
               {'function': 'lin',
                'errorDynamics':  np.zeros(dim_mp_state),
                'goal': np.ones(dim_mp_state)
               }]

        Example
        mps = [{'function': 'msd_nlin',
                'attractorStiffness': 100.,
                'mass': 1.,
                'damping': 50.,
                'goal': state_des[dim_mp_state]
               },
               {'function': 'msd',
                'attractorStiffness': 100.,
                'mass': 1.,
                'damping': 50.,
                'goal': np.ones(dim_mp_state)
                }]
        """
        Serializable._init(self, locals())

        # Define the movement primitives
        if mps is None:
            if position_mps:
                mps = [
                    {
                        'function': 'msd_nlin',
                        'attractorStiffness': 30.,
                        'mass': 1.,
                        'damping': 50.,
                        'goal': np.array([-0.8, 0.8]),  # position of the left sphere
                    },
                    {
                        'function': 'msd_nlin',
                        'attractorStiffness': 30.,
                        'mass': 1.,
                        'damping': 50.,
                        'goal': np.array([+0.8, 0.8]),  # position of the lower right sphere
                    },
                    {
                        'function': 'msd_nlin',
                        'attractorStiffness': 30.,
                        'mass': 1.,
                        'damping': 50.,
                        'goal': np.array([-0.25, 1.2]),  # position of the upper right sphere
                    }
                ]
            else:
                dt = kwargs.get('dt', 0.01)  # 100 Hz is the default
                mps = [
                    {
                        'function': 'lin',
                        'errorDynamics': 5.*np.eye(2),
                        'goal': dt*np.array([0.06, 0.06])  # X and Z [m/s]
                    },
                    {
                        'function': 'lin',
                        'errorDynamics': 5.*np.eye(2),
                        'goal': dt*np.array([-0.04, -0.04])  # X and Z [m/s]
                    }
                ]

        # Forward to the Planar3LinkSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict() if task_args is None else task_args,
            actionModelType='ds_activation',
            tasks=mps,
            **kwargs
        )

        # # State space definition
        # if kwargs.get('observeVelocities', True):
        #     self.state_mask = self.obs_space.create_mask(
        #         'Effector_X', 'Effector_Z', 'Effector_Xd', 'Effector_Zd',
        #     )
        # else:
        #     self.state_mask = self.obs_space.create_mask(
        #         'Effector_X', 'Effector_Z'
        #     )
