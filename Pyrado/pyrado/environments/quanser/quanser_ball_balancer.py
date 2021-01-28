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
from init_args_serializer import Serializable

import pyrado
from pyrado.environments.quanser import max_act_qbb
from pyrado.environments.quanser.base import QuanserReal
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ScaledExpQuadrErrRewFcn


class QBallBalancerReal(QuanserReal, Serializable):
    """ Class for the real Quanser Ball-Balancer """

    name: str = "qbb"

    def __init__(
        self,
        dt: float = 1 / 500.0,
        max_steps: int = pyrado.inf,
        task_args: [dict, None] = None,
        ip: str = "192.168.2.5",
    ):
        """
        Constructor

        :param dt: sampling frequency on the [Hz]
        :param max_steps: maximum number of steps executed on the device [-]
        :param task_args: arguments for the task construction
        :param ip: IP address of the 2 DOF Ball-Balancer platform
        """
        Serializable._init(self, locals())

        # Initialize spaces, dt, max_step, and communication
        super().__init__(ip, rcv_dim=8, snd_dim=2, dt=dt, max_steps=max_steps, task_args=task_args)
        self._curr_act = np.zeros(self.act_space.shape)  # just for usage in render function

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array(
            [np.pi / 4.0, np.pi / 4.0, 0.275 / 2.0, 0.275 / 2.0, 5 * np.pi, 5 * np.pi, 0.5, 0.5]
        )  # [rad, rad, m, m, rad/s, rad/s, rad/s, m/s, m/s]
        self._state_space = BoxSpace(
            -max_state,
            max_state,
            labels=["theta_x", "theta_y", "x", "y", "theta_dot_x", "theta_dot_y", "x_dot", "y_dot"],
        )
        self._obs_space = self._state_space
        self._act_space = BoxSpace(-max_act_qbb, max_act_qbb, labels=["V_x", "V_y"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.zeros(8))
        Q = task_args.get("Q", np.diag([1e0, 1e0, 5e3, 5e3, 1e-2, 1e-2, 5e-1, 5e-1]))
        R = task_args.get("R", np.diag([1e-2, 1e-2]))
        # Q = np.diag([1e2, 1e2, 5e2, 5e2, 1e-2, 1e-2, 1e+1, 1e+1])  # for LQR
        # R = np.diag([1e-2, 1e-2])  # for LQR

        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self.state_space, self.act_space, min_rew=1e-4)
        )

    def reset(self, *args, **kwargs) -> np.ndarray:
        # Reset socket and task
        super().reset()

        # Start with a zero action and get the first sensor measurements
        self.state = self._qsoc.snd_rcv(np.zeros(self.act_space.shape))

        # Reset time counter
        self._curr_step = 0

        return self.observe(self.state)
