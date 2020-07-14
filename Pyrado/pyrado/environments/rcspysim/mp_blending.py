import numpy as np
import os.path as osp
from init_args_serializer import Serializable
from typing import Sequence

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.goalless import GoallessTask
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
                 **kwargs):
        """
        Constructor

        :param action_model_type: `ds_activation` or `ik_activation`
        :param mps: movement primitives holding the dynamical systems and the goal states
        :param task_args: arguments for the task construction
        :param kwargs: keyword arguments forwarded to `RcsSim`
                       positionTasks: bool = True,
                       taskCombinationMethod: str = 'sum', # or 'mean', 'softmax', 'product'
        """
        Serializable._init(self, locals())

        # Forward to RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType='MPBlending',
            graphFileName='gMPBlending.xml',
            actionModelType=action_model_type,
            tasks=mps,
            positionTasks=kwargs.pop('positionTasks', True),
            **kwargs
        )

        # Store environment specific vars
        center_init_state = np.array([0., 0.])  # [m]
        self._init_space = SingularStateSpace(center_init_state, labels=['$x$', '$y$'])

    def _create_task(self, task_args: dict) -> GoallessTask:
        # Dummy task
        return GoallessTask(self.spec, ZeroPerStepRewFcn())

    @classmethod
    def get_nominal_domain_param(cls):
        raise NotImplementedError
