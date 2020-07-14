import os.path as osp
import numpy as np
from init_args_serializer import Serializable

import rcsenv
import pyrado
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import ScaledExpQuadrErrRewFcn
from pyrado.tasks.desired_state import DesStateTask
from pyrado.spaces.polar import Polar2DPosSpace


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'BallOnPlate'))


class BallOnPlateSim(RcsSim, Serializable):
    """ Base class for the ball-on-plate environments simulated in Rcs using the Vortex or Bullet physics engine """

    def __init__(self,
                 task_args: dict,
                 init_ball_vel: np.ndarray = None,
                 max_dist_force: float = None,
                 **kwargs):
        """
        Constructor

        .. note::
            This constructor should only be called via the subclasses.

        :param task_args: arguments for the task construction
        :param init_ball_vel: initial ball velocity applied to ball on `reset()`
        :param max_dist_force: maximum disturbance force, set to `None` (default) for no disturbance
        :param kwargs: keyword arguments forwarded to the `BallOnPlateSim` constructor
        """
        if init_ball_vel is not None:
            if not init_ball_vel.size() == 2:
                raise pyrado.ShapeErr(given=init_ball_vel, expected_match=np.empty(2))

        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor
        RcsSim.__init__(self,
                        envType='BallOnPlate',
                        graphFileName='gBotKuka.xml',
                        physicsConfigFile='pBallOnPlate.xml',
                        task_args=task_args,
                        **kwargs)

        # Store BallOnPlateSim specific vars
        self._init_ball_vel = init_ball_vel
        l_plate = 0.5  # [m], see the config XML-file
        min_init_state = np.array([0.7*l_plate/2, -np.pi])
        max_init_state = np.array([0.8*l_plate/2, np.pi])
        self._init_space = Polar2DPosSpace(min_init_state, max_init_state, labels=['$r$', r'$\phi$'])

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _create_task(self, task_args: dict) -> Task:
        # Needs to implemented by subclasses
        raise NotImplementedError

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(ball_mass=0.2,
                    ball_radius=0.05,
                    ball_com_x=0.,
                    ball_com_y=0.,
                    ball_com_z=0.,
                    ball_friction_coefficient=0.3,
                    ball_rolling_friction_coefficient=0.05,
                    ball_slip=50.0,
                    ball_linearvelocitydamping=0.,
                    ball_angularvelocitydamping=0.)

    def _adapt_domain_param(self, params: dict) -> dict:
        if 'ball_rolling_friction_coefficient' in params:
            br = params.get('ball_radius', None)
            if br is None:
                # If not set, get from the current simulation parameters
                br = self._impl.domainParam['ball_radius']
            return dict(params, ball_rolling_friction_coefficient=params['ball_rolling_friction_coefficient']*br)

        return params

    def _unadapt_domain_param(self, params: dict) -> dict:
        if 'ball_rolling_friction_coefficient' in params and 'ball_radius' in params:
            return dict(
                params,
                ball_rolling_friction_coefficient=params['ball_rolling_friction_coefficient']/params['ball_radius']
            )

        return params

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Call the parent class
        obs = RcsSim.reset(self, init_state, domain_param)

        # Apply a initial ball velocity if given
        if self._init_ball_vel is not None:
            self._impl.applyBallVelocity(self._init_ball_vel)
            # We could try to adapt obs here, but it's not really necessary

        return obs

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        # Sample angle and force uniformly
        angle = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(0, self._max_dist_force)
        return np.array([force*np.sin(angle), force*np.cos(angle), 0])


class BallOnPlate2DSim(BallOnPlateSim, Serializable):
    """ Ball-on-plate environment with 2-dim actions """

    name: str = 'bop2d'

    def __init__(self,
                 init_ball_vel: np.ndarray = None,
                 state_des: np.ndarray = None,
                 **kwargs):
        """
        Constructor

        :param init_ball_vel: initial ball velocity applied to ball on `reset()`
        :param state_des: desired state for the task
        :param kwargs: keyword arguments forwarded to the `BallOnPlateSim` constructor
        """
        Serializable._init(self, locals())

        # Forward to the BallOnPlateSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict(state_des=state_des),
            initBallVel=init_ball_vel,
            actionModelType='plate_angacc',
            **kwargs
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', None)
        if state_des is None:
            state_des = np.zeros(self.obs_space.flat_dim)

        Q = np.diag([1e-1, 1e-1, 1e+1, 1e+1, 0,  # Pa, Pb, Bx, By, Bz,
                     1e-3, 1e-3, 1e-2, 1e-2, 0])  # Pad, Pbd, Bxd, Byd, Bzd
        R = np.diag([1e-3, 1e-3])  # Padd, Pbdd
        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self.state_space, self.act_space, min_rew=1e-4)
        )


class BallOnPlate5DSim(BallOnPlateSim, Serializable):
    """ Ball-on-plate environment with 5-dim actions """

    name: str = 'bop5d'

    def __init__(self,
                 init_ball_vel: np.ndarray = None,
                 state_des: np.ndarray = None,
                 **kwargs):
        """
        Constructor

        :param init_ball_vel: initial ball velocity applied to ball on `reset()`
        :param state_des: desired state for the task
        :param kwargs: keyword arguments forwarded to the `RcsSim` constructor
        """
        Serializable._init(self, locals())

        # Forward to the BallOnPlateSim's constructor, specifying the characteristic action model
        super().__init__(
            task_args=dict(state_des=state_des),
            initBallVel=init_ball_vel,
            actionModelType='plate_acc5d',
            **kwargs
        )

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', None)
        if state_des is None:
            state_des = np.zeros(self.obs_space.flat_dim)
        Q = np.diag([1e-0, 1e-0, 1e-0, 1e-0, 1e-0, 1e+3, 1e+3, 1e+3,  # Px, Py, Pz, Pa, Pb, Bx, By, Bz,
                     1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-0, 1e-0, 1e-0])  # Pxd, Pyd, Pzd, Pad, Pbd, Bxd, Byd, Bzd
        R = np.diag([1e-2, 1e-2, 1e-2, 1e-3, 1e-3])  # Pxdd, Pydd, Pzdd, Padd, Pbdd
        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self.state_space, self.act_space, min_rew=1e-4)
        )
