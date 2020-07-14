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
from pyrado.tasks.masked import MaskedTask
from pyrado.tasks.reward_functions import MinusOnePerStepRewFcn, AbsErrRewFcn
from pyrado.tasks.parallel import ParallelTasks
from pyrado.tasks.predefined import create_check_all_boundaries_task, \
    create_task_space_discrepancy_task, create_collision_task
from pyrado.utils.data_types import EnvSpec


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)


def create_extract_ball_task(env_spec: EnvSpec, continuous_rew_fcn):
    # Define the indices for selection. This needs to match the observations' names in RcsPySim.
    idcs = ['Ball_X', 'Ball_Y']

    # Get the masked environment specification
    spec = EnvSpec(
        env_spec.obs_space,
        env_spec.act_space,
        env_spec.state_space.subspace(env_spec.state_space.create_mask(idcs))
    )

    # Create a desired state task
    des_state = np.array([1., 0])
    if continuous_rew_fcn:
        q = np.array([0./np.pi])
        r = 1e-6*np.ones(spec.act_space.flat_dim)
        rew_fcn = AbsErrRewFcn(q, r)
    else:
        rew_fcn = MinusOnePerStepRewFcn()
    dst_task = DesStateTask(spec, des_state, rew_fcn)

    # Return the masked tasks
    return MaskedTask(env_spec, dst_task, idcs)


class BallInTubeSim(RcsSim, Serializable):
    """ Base class for 2-armed humanoid robot fiddling a ball out of a tube """

    def __init__(self,
                 task_args: dict,
                 ref_frame: str,
                 position_mps: bool,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None],
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param fixed_init_state: use an init state space with only one state (e.g. for debugging)
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
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
            envType='BallInTube',
            physicsConfigFile='pBallInTube.xml',
            extraConfigDir=osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'BallInTube'),
            hudColor='BLACK_RUBBER',
            task_args=task_args,
            refFrame=ref_frame,
            positionTasks=position_mps,
            tasksLeft=mps_left,
            tasksRight=mps_right,
            **kwargs
        )

        # Initial state space definition
        if fixed_init_state:
            dafault_init_state = np.array([-0.2, 0., 0., 0.85])  # [m, m, rad, m]
            self._init_space = SingularStateSpace(dafault_init_state,
                                                  labels=['$x$', '$y$', '$\theta$', '$z$'])
        else:
            min_init_state = np.array([0.05, -0.05, -5*np.pi/180, 0.8])
            max_init_state = np.array([0.25, 0.05, 5*np.pi/180, 0.9])
            self._init_space = BoxSpace(min_init_state, max_init_state,  # [m, m, rad, m]
                                        labels=['$x$', '$y$', '$\theta$', '$z$'])

    def _create_task(self, task_args: dict) -> Task:
        # Create the tasks
        continuous_rew_fcn = task_args.get('continuous_rew_fcn', True)
        # task_box = create_extract_ball_task(self.spec, continuous_rew_fcn)
        task_check_bounds = create_check_all_boundaries_task(self.spec, penalty=1e3)
        # task_collision = create_collision_task(self.spec, factor=1.)
        # task_ts_discrepancy = create_task_space_discrepancy_task(self.spec,
        #                                                          AbsErrRewFcn(q=0.5*np.ones(3),
        #                                                                       r=np.zeros(self.act_space.shape)))

        return ParallelTasks([
            # task_box,
            task_check_bounds,
            # task_collision,
            # task_ts_discrepancy
        ], hold_rew_when_done=False)

    @classmethod
    def get_nominal_domain_param(cls):
        return dict(ball_mass=0.3,
                    ball_radius=0.03,
                    ball_rolling_friction_coefficient=0.05,
                    slider_mass=0.3, )
        # table_friction_coefficient=0.6)


class BallInTubePosMPsSim(BallInTubeSim, Serializable):
    """ Humanoid robot fiddling a ball out of a tube using two hooks and position-level movement primitives """

    name: str = 'bit-pos'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None],
                 continuous_rew_fcn: bool = True,
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param fixed_init_state: use an init state space with only one state (e.g. for debugging)
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
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
        if mps_left is None:
            mps_left = [
                # Effector position relative to slider
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([-0.05, 0.05, 0.05])},  # [m]
                # Effector orientation relative to slider
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([0., 0., 0.])},  # [rad]
            ]
        if mps_right is None:
            mps_right = [
                # Effector position relative to slider
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([0.05, -0.05, 0.05])},  # [m]
                # Effector orientation relative to slider
                {'function': 'msd_nlin', 'attractorStiffness': 30., 'mass': 1., 'damping': 60.,
                 'goal': np.array([0., 0., 0.])},  # [rad]
            ]

        # Forward to the BallInTubeSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            position_mps=True,
            mps_left=mps_left,
            mps_right=mps_right,
            fixed_init_state=fixed_init_state,
            **kwargs
        )


class BallInTubeVelMPsSim(BallInTubeSim, Serializable):
    """ Humanoid robot fiddling a ball out of a tube using two hooks and velocity-level movement primitives """

    name: str = 'bit-vel'

    def __init__(self,
                 ref_frame: str,
                 mps_left: [Sequence[dict], None],
                 mps_right: [Sequence[dict], None],
                 continuous_rew_fcn: bool = True,
                 fixed_init_state: bool = False,
                 **kwargs):
        """
        Constructor

        :param ref_frame: reference frame for the MPs, e.g. 'world', or 'table'
        :param mps_left: left arm's movement primitives holding the dynamical systems and the goal states
        :param mps_right: right arm's movement primitives holding the dynamical systems and the goal states
        :param continuous_rew_fcn: specify if the continuous or an uninformative reward function should be used
        :param fixed_init_state: use an init state space with only one state (e.g. for debugging)
        :param kwargs: keyword arguments which are available for all task-based `RcsSim`
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
        dt = kwargs.get('dt', 0.01)  # 100 Hz is the default
        # basket_extends = self.get_body_extents('Basket', 0)
        if mps_left is None:
            mps_left = [
                # Effector Xd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Effector Yd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Effector Zd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Effector Ad
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Effector Bd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Effector Cd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
            ]
        if mps_right is None:
            mps_right = [
                # Effector Xd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Effector Yd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Effector Zd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([0.15])},  # [m/s]
                # Effector Ad
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Effector Bd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
                # Effector Cd
                {'function': 'lin', 'errorDynamics': 1., 'goal': dt*np.array([15/180*np.pi])},  # [rad/s]
            ]

        # Forward to the BallInTubeSim's constructor
        super().__init__(
            task_args=dict(continuous_rew_fcn=continuous_rew_fcn),
            ref_frame=ref_frame,
            position_mps=False,
            mps_left=mps_left,
            mps_right=mps_right,
            fixed_init_state=fixed_init_state,
            **kwargs
        )
