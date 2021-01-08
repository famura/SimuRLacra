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

import rcsenv
from pyrado.environments.rcspysim.base import RcsSim
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn


rcsenv.addResourcePath(rcsenv.RCSPYSIM_CONFIG_PATH)
rcsenv.addResourcePath(osp.join(rcsenv.RCSPYSIM_CONFIG_PATH, "QuanserQube"))


class QQubeRcsSim(RcsSim, Serializable):
    """
    Swing-up task on the underactuated Quanser Qube a.k.a. Furuta pendulum, simulated in Rcs

    .. note::
        The action is different to the `QQubeSim` in the directory `sim_py`.
    """

    name: str = "qq-rcs"

    def __init__(self, task_args: [dict, None] = None, max_dist_force: [float, None] = None, **kwargs):
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
            envType="QuanserQube",
            graphFileName="gQuanserQube_trqCtrl.xml",
            physicsConfigFile="pQuanserQube.xml",
            actionModelType="joint_acc",
            **kwargs,
        )

        # Setup disturbance
        self._max_dist_force = max_dist_force

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([2e-1, 1.0, 2e-2, 5e-3]))
        R = task_args.get("R", np.diag([3e-3]))
        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def _disturbance_generator(self) -> (np.ndarray, None):
        if self._max_dist_force is None:
            return None
        else:
            # Sample angle and force uniformly
            angle = np.random.uniform(-np.pi, np.pi)
            force = np.random.uniform(0, self._max_dist_force)
            return np.array([force * np.sin(angle), force * np.cos(angle), 0])
