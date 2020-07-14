import numpy as np
import os.path as osp
from init_args_serializer import Serializable

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'QuanserQube'))


class QQubeRcsSim(RcsSim, Serializable):
    """
    Swing-up task on the underactuated Quanser Qube a.k.a. Furuta pendulum, simulated in Rcs

    .. note::
        The action is different to the `QQubeSim` in the directory `sim_py`.
    """

    name: str = 'qq-rcs'

    def __init__(self,
                 task_args: [dict, None] = None,
                 max_dist_force: [float, None] = None,
                 **kwargs):
        """
        Constructor

        :param task_args: arguments for the task construction, pass `None` to use default values
                          state_des: 4-dim np.ndarray
                          Q: 4x4-dim np.ndarray
                          R: 4x4-dim np.ndarray
        :param max_dist_force: maximum disturbance force, pass `None` for no disturbance
        :param kwargs: keyword arguments forwarded to the RcsSim
        """
        Serializable._init(self, locals())

        # Forward to RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args if task_args is not None else dict(),
            envType='QuanserQube',
            graphFileName='gQuanserQube_trqCtrl.xml',
            physicsConfigFile='pQuanserQube.xml',
            actionModelType='joint_acc',
            **kwargs
        )

        # Store QQubeRcsSim specific vars
        max_init_state = np.array([5., 3., 0.5, 0.5])/180*np.pi  # [rad, rad, rad/s, rad/s]
        self._init_space = BoxSpace(-max_init_state, max_init_state,
                                    labels=[r'$\theta$', r'$\alpha$', r'$\dot{\theta}$', r'$\dot{\alpha}$'])

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', np.array([0., np.pi, 0., 0.]))
        Q = task_args.get('Q', np.diag([2e-1, 1., 2e-2, 5e-3]))
        R = task_args.get('R', np.diag([3e-3]))
        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        else:
            # Sample angle and force uniformly
            angle = np.random.uniform(-np.pi, np.pi)
            force = np.random.uniform(0, self._max_dist_force)
            return np.array([force*np.sin(angle), force*np.cos(angle), 0])
