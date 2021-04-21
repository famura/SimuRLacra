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
from typing import Sequence

import numpy as np
import rcsenv
from init_args_serializer import Serializable

from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.goalless import GoallessTask
from pyrado.tasks.reward_functions import ZeroPerStepRewFcn


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "MPBlending"))


class MPBlendingSim(RcsSim, Serializable):
    """ Sandbox class for testing different ways of combining a set of movement primitives """

    name: str = "mpb"

    def __init__(self, action_model_type: str, mps: Sequence[dict] = None, task_args: dict = None, **kwargs):
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

        task_spec_ik = [
            dict(x_des=np.array([0.0, 0.0, 0.0])),
            dict(x_des=np.array([0.0, 0.0, 0.0])),
            dict(x_des=np.array([0.0, 0.0, 0.0])),
            dict(x_des=np.array([0.0, 0.0, 0.0])),
        ]

        # Forward to RcsSim's constructor
        RcsSim.__init__(
            self,
            task_args=task_args,
            envType="MPBlending",
            graphFileName="gMPBlending.xml",
            actionModelType=action_model_type,
            tasks=mps,
            positionTasks=kwargs.pop("positionTasks", None),  # invalid default value, positionTasks can be unnecessary
            taskSpecIK=task_spec_ik,
            **kwargs,
        )

    def _create_task(self, task_args: dict) -> GoallessTask:
        # Dummy task
        return GoallessTask(self.spec, ZeroPerStepRewFcn())

    @classmethod
    def get_nominal_domain_param(cls):
        raise NotImplementedError
