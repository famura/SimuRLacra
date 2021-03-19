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
from init_args_serializer.serializable import Serializable

from pyrado.environments.pysim.base import SimPyEnv
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.spaces.compound import CompoundSpace
from pyrado.tasks.base import Task
from pyrado.tasks.reward_functions import ScaledExpQuadrErrRewFcn
from pyrado.tasks.desired_state import DesStateTask


class BallOnBeamSim(SimPyEnv, Serializable):
    """
    Environment in which a ball rolls on a beam (1 dim). The ball is randomly initialized on the beam and is to be
    stabilized on the center of the beam. In this setup, the agent can control the torque applied to the beam.
    """

    name: str = "bob"

    def _create_spaces(self):
        l_beam = self.domain_param["l_beam"]
        g = self.domain_param["g"]

        # Set the bounds for the system's states and actions
        max_state = np.array([l_beam / 2.0, np.pi / 4.0, 10.0, np.pi])
        max_act = np.array([l_beam / 2.0 * g * 3.0])  # max torque [Nm]; the factor 3.0 is arbitrary
        self._curr_act = np.zeros_like(max_act)  # just for usage in render function

        self._state_space = BoxSpace(-max_state, max_state, labels=["x", "alpha", "x_dot", "alpha_dot"])
        self._obs_space = self._state_space
        self._init_space = CompoundSpace(
            [
                BoxSpace(
                    np.array([-0.8 * l_beam / 2.0, -5 / 180.0 * np.pi, -0.02 * max_state[2], -0.02 * max_state[3]]),
                    np.array([-0.7 * l_beam / 2.0, +5 / 180.0 * np.pi, +0.02 * max_state[2], +0.02 * max_state[3]]),
                    labels=["x", "alpha", "x_dot", "alpha_dot"],
                ),
                BoxSpace(
                    np.array([0.7 * l_beam / 2.0, -5 / 180.0 * np.pi, -0.02 * max_state[2], -0.02 * max_state[3]]),
                    np.array([0.8 * l_beam / 2.0, +5 / 180.0 * np.pi, +0.02 * max_state[2], +0.02 * max_state[3]]),
                    labels=["x", "alpha", "x_dot", "alpha_dot"],
                ),
            ]
        )
        self._act_space = BoxSpace(-max_act, max_act, labels=["tau"])

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            g=9.81,  # gravity constant [m/s**2]
            m_ball=0.5,  # ball mass [kg]
            r_ball=0.1,  # ball radius [m]
            m_beam=3.0,  # beam mass [kg]
            l_beam=2.0,  # beam length [m]
            d_beam=0.1,  # beam thickness [m]
            c_frict=0.05,  # viscous friction coefficient [Ns/m]
            ang_offset=0.0,
        )  # constant beam angle offset [rad]

    def _calc_constants(self):
        m_ball = self.domain_param["m_ball"]
        r_ball = self.domain_param["r_ball"]
        m_beam = self.domain_param["m_beam"]
        l_beam = self.domain_param["l_beam"]
        d_beam = self.domain_param["d_beam"]

        self.J_ball = 2.0 / 5 * m_ball * r_ball ** 2
        self.J_beam = 1.0 / 12 * m_beam * (l_beam ** 2 + d_beam ** 2)
        self.zeta_ball = m_ball + self.J_ball / r_ball ** 2

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.zeros(4))
        Q = task_args.get("Q", np.diag([1e5, 1e3, 1e3, 1e2]))
        R = task_args.get("R", np.eye(self.spec.act_space.flat_dim))

        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self.state_space, self.act_space, min_rew=1e-4)
        )

    def _step_dynamics(self, act: np.ndarray):
        g = self.domain_param["g"]
        m_ball = self.domain_param["m_ball"]
        c_frict = self.domain_param["c_frict"]
        ang_offset = self.domain_param["ang_offset"]

        # Nonlinear dynamics
        x = self.state[0]  # ball position
        a = self.state[1] + ang_offset  # beam angular position
        x_dot = self.state[2]  # ball velocity
        a_dot = self.state[3]  # beam angular velocity
        zeta_beam = m_ball * x ** 2 + self.J_beam  # depends on the bal position

        # EoM solved for the accelerations
        x_ddot = (-c_frict * x_dot + m_ball * x * a_dot ** 2 - m_ball * g * np.sin(a)) / self.zeta_ball
        a_ddot = (float(act) - 2.0 * m_ball * x * x_dot * a_dot - m_ball * g * np.cos(a) * x) / zeta_beam

        # Integration step (symplectic Euler)
        self.state[2:] += np.array([x_ddot, a_ddot]) * self._dt  # next velocity
        self.state[:2] += self.state[2:] * self._dt  # next position

    def _init_anim(self):
        # Import PandaVis Class
        from pyrado.environments.pysim.pandavis import BallOnBeamVis

        # Create instance of PandaVis
        self._visualization = BallOnBeamVis(self, self._rendering)


class BallOnBeamDiscSim(BallOnBeamSim, Serializable):
    """ Ball-on-beam simulation environment with discrete actions """

    name: str = "bob-d"

    def __init__(self, *args, **kwargs):
        """
        Constructor

        :param args: forwarded to BallOnBeamSim's constructor
        :param kwargs: forwarded to BallOnBeamSim's constructor
        """
        Serializable._init(self, locals())
        super().__init__(*args, **kwargs)

    def _create_spaces(self):
        super()._create_spaces()

        # Define discrete action space form the specification of the continuous
        min_act, max_act = self._act_space.bounds
        num_act = 3
        linspaced = np.linspace(min_act, max_act, num=num_act, endpoint=True)
        self._act_space = DiscreteSpace(linspaced, labels=["tau"])
