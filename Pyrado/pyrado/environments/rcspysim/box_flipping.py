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
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.endless_flipping import EndlessFlippingTask
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn, MinusOnePerStepRewFcn, AbsErrRewFcn, CosOfOneEleRewFcn, \
    CompoundRewFcn, RewFcn
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.utils import proximity_succeeded, never_succeeded
from pyrado.tasks.predefined import create_check_all_boundaries_task, create_collision_task
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)


def create_task_space_discrepancy_task(env_spec: EnvSpec, rew_fcn: RewFcn) -> MaskedTask:
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ['DiscrepTS_Y', 'DiscrepTS_Z']

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task (no task space discrepancy is desired and the task never stops because of success)
    dst = DesStateTask(spec, np.zeros(spec.state_space.shape), rew_fcn, never_succeeded)

    # Mask selected discrepancy observation
    return MaskedTask(env_spec, dst, idcs)


def create_box_flip_task(env_spec: EnvSpec, continuous_rew_fcn):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ['Box_A']

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    # state_des = np.array([0.3])  # box position is measured relative to the table
    state_des = np.array([-np.pi/2])  # box position is measured world coordinates
    if continuous_rew_fcn:
        # q = np.array([0./np.pi])
        # r = 1e-6*np.ones(spec.act_space.flat_dim)
        # rew_fcn_act = AbsErrRewFcn(q, r)
        rew_fcn = CosOfOneEleRewFcn(idx=0)
        # rew_fcn = CompoundRewFcn([rew_fcn_act, rew_fcn_box])
    else:
        rew_fcn = MinusOnePerStepRewFcn()
    ef_task = EndlessFlippingTask(spec, rew_fcn, init_angle=0.)

    # Return the masked tasks
    return MaskedTask(env_spec, ef_task, idcs)


class BoxFlippingSim(RcsSim, Serializable):
    """ Base class for simplified robotic manipulator flipping a box over and over again """

    def __init__(self,
                 task_args: dict,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None] = None,
                 mps_right: [Sequence[dict], None] = None,
                 **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param ref_frame: reference frame for the MPs, e.g. 'world', 'table', or 'box'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       positionTasks: bool,
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
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

        # Forward to the RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType='BoxFlipping',
            physicsConfigFile='pBoxFlipping.xml',
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'BoxFlipping'),
            hudColor='BLACK_RUBBER',
            refFrame=ref_frame,
            positionTasks=kwargs.pop('positionTasks'),
            tasksLeft=mps_left,
            tasksRight=mps_right,
            **kwargs
        )

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        continuous_rew_fcn = task_args.get('continuous_rew_fcn', True)
        task_box = create_box_flip_task(self.spec, continuous_rew_fcn)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        # task_collision = create_collision_task(self.spec, factor=1e-2)
        # task_ts_discrepancy = create_task_space_discrepancy_task(self.spec,
        #                                                          AbsErrRewFcn(q=1e-2*np.ones(2),
        #                                                                       r=np.zeros(self.act_space.shape)))

        return ParallelTasks([
            task_box,
            task_check_bounds,
            # task_collision,
            # task_ts_discrepancy
        ], hold_rew_when_done=False)

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(box_length=0.18,
                    box_width=0.14,
                    box_mass=0.3,
                    box_friction_coefficient=1.4,
                    table_friction_coefficient=1.0)


class BoxFlippingIKActivationSim(BoxFlippingSim, Serializable):
    """ Simplified robotic manipulator flipping a box over and over again using a Rcs IK-based controller """

    name: str = 'bf-ika'

    def __init__(self, ref_frame: str, continuous_rew_fcn: bool = True, **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the Rcs tasks, e.g. 'world', 'table', or 'box'
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
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
            dict(x_des=np.array([0.])),  # Y left
            dict(x_des=np.array([0.])),  # Z left
            dict(x_des=np.array([0.])),  # Y left
            dict(x_des=np.array([0.])),  # Z left
        ]

        # Forward to the BoxFlippingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            actionModelType='ik_activation',
            taskSpecIK=task_spec_ik,
            **kwargs
        )


class BoxFlippingPosDSSim(BoxFlippingSim, Serializable):
    """ Simplified robotic manipulator flipping a box over and over again using position-level movement primitives """

    name: str = 'bf-pos'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None] = None,
                 continuous_rew_fcn: bool = True,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'table', or 'box'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'sum', # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
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

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        if mps_left is None:
            mps_left = [
                # Y
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([-0.8])},  # [m]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([+0.8])},  # [m]
                # Z
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([-0.0])},  # [m]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([+0.2])},  # [m]
            ]
        if mps_right is None:
            mps_right = [
                # Y
                # {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                #  'goal': np.array([-0.8])},  # [m]
                # {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                #  'goal': np.array([+0.8])},  # [m]
                # # Z
                # {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                #  'goal': np.array([-0.0])},  # [m]
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([+0.2])},  # [m]
                # Distance
                # {'function': 'msd', 'attractorStiffness': 50., 'mass': 1., 'damping': 10.,
                {'function': 'lin', 'errorDynamics': 1.,  # [m/s]
                 'goal': np.array([0.0])},  # [m]
            ]

        # Forward to the BoxFlippingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            mps_left=mps_left,
            mps_right=mps_right,
            actionModelType='ds_activation',
            positionTasks=True,
            **kwargs
        )


class BoxFlippingVelDSSim(BoxFlippingSim, Serializable):
    """ Simplified robotic manipulator flipping a box over and over again using velocity-level movement primitives """

    name: str = 'bf-vel'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None],
                 continuous_rew_fcn: bool = True,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', 'table', or 'box'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
                       taskCombinationMethod: str = 'sum', # or 'mean', 'softmax', 'product'
                       checkJointLimits: bool = False,
                       collisionAvoidanceIK: bool = True,
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

        # Fall back to some defaults of no MPs are defined (e.g. for testing)
        dt = kwargs.get('dt', 0.01)  # 100 Hz is the default
        # basket_extends = self.get_body_extents('Basket', 0)
        if mps_left is None:
            mps_left = [
                # Yd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.1])},  # [m/s]
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.1])},  # [m/s]
                # Zd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.1])},  # [m/s]
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.1])},  # [m/s]
            ]
        if mps_right is None:
            mps_right = [
                # Yd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.1])},  # [m/s]
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.1])},  # [m/s]
                # Zd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.1])},  # [m/s]
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([-0.1])},  # [m/s]
            ]

        # Forward to the BoxFlippingSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            mps_left=mps_left,
            mps_right=mps_right,
            actionModelType='ds_activation',
            positionTasks=False,
            **kwargs
        )
