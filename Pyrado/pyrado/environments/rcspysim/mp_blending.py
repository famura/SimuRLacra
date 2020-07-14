import numpy as np
import os.path as osp
from typing import Sequence
from init_args_serializer import Serializable

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, 'MPBlending'))


class MPBlendingSim(RcsSim, Serializable):
    """ Sandbox class for testing different ways of combining a set of movement primitives """

    name: str = 'mpb'

    def __init__(self,
                 action_model_type: str,
                 mps: Sequence[dict] = None,
                 task_args: dict = None,
                 max_dist_force: float = None,
                 **kwargs):
        """
        Constructor

        :param action_model_type: `ds_activation` or `ik_activation`
        :param mps: movement primitives holding the dynamical systems and the goal states
        :param task_args: arguments for the task construction
        :param max_dist_force: maximum disturbance force, set to None (default) for no disturbance
        :param position_mps: `True` if the MPs are defined on position level, `False` if defined on velocity level,
                             only matters if `actionModelType='ds_activation'`
        :param kwargs: keyword arguments forwarded to `RcsSim`
                       positionTasks: bool = True,
                       taskCombinationMethod: str = 'sum', # or 'mean', 'softmax', 'product'
        """
        Serializable._init(self, locals())

        # Forward to the RcsSim's constructor, nothing more needs to be done here
        RcsSim.__init__(
            self,
            envType='MPBlending',
            task_args=task_args,
            actionModelType=action_model_type,
            graphFileName='gMPBlending.xml',
            tasks=mps,
            positionTasks=kwargs.pop('positionTasks', True),
            **kwargs
        )

        # Store Planar3Link specific vars
        center_init_state = np.array([0., 0.])  # [m]
        self._init_space = SingularStateSpace(center_init_state, labels=['$x$', '$y$'])

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _create_task(self, task_args: dict) -> DesStateTask:
        # Dummy task
        return DesStateTask(self.spec, np.zeros(self.state_space.shape), ZeroPerStepRewFcn())

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        # Sample angle and force uniformly
        angle = np.random.uniform(-np.pi, np.pi)
        force = np.random.uniform(0, self._max_dist_force)
        return np.array([force*np.cos(angle), force*np.sin(angle), 0])

    @classmethod
    def get_nominal_domain_param(cls):
        raise NotImplementedError
