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

import pathlib
import numpy as np
from init_args_serializer.serializable import Serializable

from pyrado.environments.pysim.base import SimPyEnv
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn

class PendulumSim(SimPyEnv, Serializable):
    """ Under-actuated inverted pendulum environment similar to the one from OpenAI Gym """

    name: str = "pend"

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([4 * np.pi, 4 * np.pi])  # [rad, rad/s]
        max_obs = np.array([1.0, 1.0, np.inf])  # [-, -, rad/s]
        init_state = np.zeros(2)  # [rad, rad/s]
        tau_max = self.domain_param["tau_max"]

        self._state_space = BoxSpace(-max_state, max_state, labels=["theta", "theta_dot"])
        self._obs_space = BoxSpace(-max_obs, max_obs, labels=["sin_theta", "cos_theta", "theta_dot"])
        self._init_space = SingularStateSpace(init_state, labels=["theta", "theta_dot"])
        self._act_space = BoxSpace(-tau_max, tau_max, labels=["tau"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([np.pi, 0.0]))
        Q = task_args.get("Q", np.diag([1e-0, 1e-3]))
        R = task_args.get("R", np.diag([1e-2]))

        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def observe(self, state) -> np.ndarray:
        return np.array([np.sin(state[0]), np.cos(state[0]), state[1]])

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            g=9.81,  # gravity constant [m/s**2]
            m_pole=1.0,  # mass of the pole [kg]
            l_pole=1.0,  # half pole length [m]
            d_pole=0.05,  # rotational damping of the pole [kg*m**2/s]
            tau_max=3.5,
        )  # maximum applicable torque [N*m] (under-actuated if < m*l*g/2)

    def _step_dynamics(self, act: np.ndarray):
        g = self.domain_param["g"]
        m_pole = self.domain_param["m_pole"]
        l_pole = self.domain_param["l_pole"]
        d_pole = self.domain_param["d_pole"]

        # Dynamics (pendulum modeled as a rod)
        th, th_dot = self.state
        th_ddot = (act - m_pole * g * l_pole / 2.0 * np.sin(th) - d_pole * th_dot) / (m_pole * l_pole ** 2 / 3.0)

        # Integration step (symplectic Euler)
        self.state[1] += th_ddot * self._dt  # next velocity
        self.state[0] += self.state[1] * self._dt  # next position

    def _init_anim(self):
        # Import PandaVis Class
        from pyrado.environments.pysim.pandavis import PendulumVis
        # Create instance of PandaVis
        self._visualization = PendulumVis(self, self._render)

