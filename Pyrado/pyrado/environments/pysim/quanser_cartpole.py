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
from abc import abstractmethod
from init_args_serializer.serializable import Serializable

import pyrado
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.quanser import max_act_qcp
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import QuadrErrRewFcn, UnderActuatedSwingUpRewFcn


class QCartPoleSim(SimPyEnv, Serializable):
    """ Base Environment for the Quanser Cart-Pole swing-up and stabilization task """

    def __init__(self,
                 dt: float,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None,
                 long: bool = False):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: long (`True`) or short (`False`) pole
        """
        Serializable._init(self, locals())

        self._obs_space = None
        self._long = long
        self.x_buffer = 0.05  # [m]

        # Call SimPyEnv's constructor
        super().__init__(dt, max_steps, task_args)

        # Update the class-specific domain parameters
        self.domain_param = self.get_nominal_domain_param(long=long)

    def _create_spaces(self):
        l_rail = self.domain_param['l_rail']
        max_obs = np.array([l_rail/2., 1., 1., np.inf, np.inf])

        self._state_space = None
        self._obs_space = BoxSpace(-max_obs, max_obs, labels=['$x$', r'$\sin(\theta)$', r'$\cos(\theta)$',
                                                              r'$\dot{x}$', r'$\dot{\theta}$'])
        self._init_space = None
        self._act_space = BoxSpace(-max_act_qcp, max_act_qcp, labels=['$V$'])

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        raise NotImplementedError

    def observe(self, state):
        return np.array([state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    @classmethod
    def get_nominal_domain_param(cls, long: bool = False) -> dict:
        if long:
            m_pole = 0.23
            l_pole = 0.641/2
        else:
            m_pole = 0.127
            l_pole = 0.3365/2

        return dict(
            g=9.81,  # gravity constant [m/s**2]
            m_cart=0.38,  # mass of the cart [kg]
            l_rail=0.814,  # length of the rail the cart is running on [m]
            eta_m=0.9,  # motor efficiency [-], default 1.
            eta_g=0.9,  # planetary gearbox efficiency [-], default 1.
            K_g=3.71,  # planetary gearbox gear ratio [-]
            J_m=3.9e-7,  # rotor inertia [kg*m**2]
            r_mp=6.35e-3,  # motor pinion radius [m]
            R_m=2.6,  # motor armature resistance [Ohm]
            k_m=7.67e-3,  # motor torque constant [N*m/A] = back-EMF constant [V*s/rad]
            B_pole=0.0024,  # viscous coefficient at the pole [N*s]
            B_eq=5.4,  # equivalent Viscous damping coefficient [N*s/m]
            m_pole=m_pole,  # mass of the pole [kg]
            l_pole=l_pole,  # half pole length [m]
        )

    def _calc_constants(self):
        l_pole = self.domain_param['l_pole']
        m_pole = self.domain_param['m_pole']
        m_cart = self.domain_param['m_cart']
        eta_g = self.domain_param['eta_g']
        K_g = self.domain_param['K_g']
        J_m = self.domain_param['J_m']
        r_mp = self.domain_param['r_mp']

        self.J_pole = l_pole**2*m_pole/3.  # pole inertia [kg*m**2]
        self.J_eq = m_cart + (eta_g*K_g**2*J_m)/r_mp**2  # equiv. inertia [kg]

    def _step_dynamics(self, act: np.ndarray):
        g = self.domain_param['g']
        l_pole = self.domain_param['l_pole']
        m_pole = self.domain_param['m_pole']
        eta_m = self.domain_param['eta_m']
        eta_g = self.domain_param['eta_g']
        K_g = self.domain_param['K_g']
        R_m = self.domain_param['R_m']
        k_m = self.domain_param['k_m']
        r_mp = self.domain_param['r_mp']
        B_eq = self.domain_param['B_eq']
        B_pole = self.domain_param['B_pole']

        x, th, x_dot, th_dot = self.state

        # Force acting on the cart
        force = (eta_g*K_g*eta_m*k_m)/(R_m*r_mp)*(eta_m*float(act) - K_g*k_m*x_dot/r_mp)

        A = np.array([[m_pole + self.J_eq, m_pole*l_pole*np.cos(th)],
                      [m_pole*l_pole*np.cos(th), self.J_pole + m_pole*l_pole**2]])

        b = np.array([force - B_eq*x_dot - m_pole*l_pole*np.sin(th)*th_dot**2,
                      - B_pole*th_dot - m_pole*l_pole*g*np.sin(th)])

        # Compute acceleration from linear system of equations
        x_ddot, theta_ddot = np.linalg.solve(A, b)

        # Integration step (symplectic Euler)
        self.state[2:] += np.array([x_ddot, theta_ddot])*self._dt  # next velocity
        self.state[:2] += self.state[2:]*self._dt  # next position

    def _init_anim(self):
        import vpython as vp

        l_pole = float(self.domain_param['l_pole'])
        l_rail = float(self.domain_param['l_rail'])

        # Only for animation
        l_cart, h_cart = 0.08, 0.08
        r_pole, r_rail = 0.01, 0.005

        # Get positions
        x, th, _, _ = self.state

        self._anim['canvas'] = vp.canvas(width=1000, height=600, title="Quanser Cartpole")
        # Rail
        self._anim['rail'] = vp.cylinder(
            pos=vp.vec(-l_rail/2, -h_cart/2 - r_rail, 0),  # a VPython's cylinder origin is at the bottom
            radius=r_rail,
            length=l_rail,
            color=vp.color.white,
            canvas=self._anim['canvas'])
        # Cart
        self._anim['cart'] = vp.box(
            pos=vp.vec(x, 0, 0),
            length=l_cart, height=h_cart, width=h_cart/2,
            color=vp.color.green,
            canvas=self._anim['canvas'])
        self._anim['joint'] = vp.sphere(
            pos=vp.vec(x, 0, r_pole + h_cart/4),
            radius=r_pole,
            color=vp.color.white,
        )
        # Pole
        self._anim['pole'] = vp.cylinder(
            pos=vp.vec(x, 0, r_pole + h_cart/4),
            axis=vp.vec(2*l_pole*vp.sin(th), -2*l_pole*vp.cos(th), 0),
            radius=r_pole,
            length=2*l_pole,
            color=vp.color.blue,
            canvas=self._anim['canvas'])

    def _update_anim(self):
        import vpython as vp

        g = self.domain_param['g']
        m_cart = self.domain_param['m_cart']
        m_pole = self.domain_param['m_pole']
        l_pole = float(self.domain_param['l_pole'])
        l_rail = float(self.domain_param['l_rail'])
        eta_m = self.domain_param['eta_m']
        eta_g = self.domain_param['eta_g']
        K_g = self.domain_param['K_g']
        J_m = self.domain_param['J_m']
        R_m = self.domain_param['R_m']
        k_m = self.domain_param['k_m']
        r_mp = self.domain_param['r_mp']
        B_eq = self.domain_param['B_eq']
        B_pole = self.domain_param['B_pole']

        # Only for animation
        l_cart, h_cart = 0.08, 0.08
        r_pole, r_rail = 0.01, 0.005

        # Get positions
        x, th, _, _ = self.state

        # Rail
        self._anim['rail'].pos = vp.vec(-l_rail/2, -h_cart/2 - r_rail, 0)
        self._anim['rail'].length = l_rail
        # Cart
        self._anim['cart'].pos = vp.vec(x, 0, 0)
        self._anim['joint'].pos = vp.vec(x, 0, r_pole + h_cart/4)
        # Pole
        self._anim['pole'].pos = vp.vec(x, 0, r_pole + h_cart/4)
        self._anim['pole'].axis = vp.vec(2*l_pole*vp.sin(th), -2*l_pole*vp.cos(th), 0)

        # Set caption text
        self._anim['canvas'].caption = f"""
                    x: {self.state[0] : 1.4f}
                    theta: {self.state[1]*180/np.pi : 2.3f}
                    dt: {self._dt :1.4f}
                    g: {g : 1.3f}
                    m_cart: {m_cart : 1.4f}
                    l_rail: {l_rail : 1.3f}
                    l_pole: {l_pole : 1.3f} (0.168 is short)
                    eta_m: {eta_m : 1.3f}
                    eta_g: {eta_g : 1.3f}
                    K_g: {K_g : 1.3f}
                    J_m: {J_m : 1.8f}
                    r_mp: {r_mp : 1.4f}
                    R_m: {R_m : 1.3f}
                    k_m: {k_m : 1.6f}
                    B_eq: {B_eq : 1.2f}
                    B_pole: {B_pole : 1.3f}
                    m_pole: {m_pole : 1.3f}
                    """


class QCartPoleStabSim(QCartPoleSim, Serializable):
    """
    Environment in which a pole has to be stabilized in the upright position (inverted pendulum) by moving a cart on
    a rail. The pole can rotate around an axis perpendicular to direction in which the cart moves.
    """

    name: str = 'qcp-st'

    def __init__(self,
                 dt: float,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None,
                 long: bool = False):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: long (`True`) or short (`False`) pole
        """
        Serializable._init(self, locals())

        self.stab_thold = 15/180.*np.pi  # threshold angle for the stabilization task to be a failure [rad]
        self.max_init_th_offset = 8/180.*np.pi  # [rad]

        super().__init__(dt, max_steps, task_args, long)

    def _create_spaces(self):
        super()._create_spaces()
        l_rail = self.domain_param['l_rail']

        min_state = np.array([-l_rail/2. + self.x_buffer, np.pi - self.stab_thold,
                              -l_rail, -2*np.pi])  # [m, rad, m/s, rad/s]
        max_state = np.array([+l_rail/2. - self.x_buffer, np.pi + self.stab_thold,
                              +l_rail, +2*np.pi])  # [m, rad, m/s, rad/s]

        max_init_state = np.array(
            [+0.05, np.pi + self.max_init_th_offset, +0.05, +8/180*np.pi])  # [m, rad, m/s, rad/s]
        min_init_state = np.array(
            [-0.05, np.pi - self.max_init_th_offset, -0.05, -8/180*np.pi])  # [m, rad, m/s, rad/s]

        self._state_space = BoxSpace(min_state, max_state,
                                     labels=['$x$', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$'])
        self._init_space = BoxSpace(min_init_state, max_init_state,
                                    labels=['$x$', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$'])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', np.array([0., np.pi, 0., 0.]))
        Q = task_args.get('Q', np.diag([5e-0, 1e+1, 1e-2, 1e-2]))
        R = task_args.get('R', np.diag([1e-3]))

        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, QuadrErrRewFcn(Q, R), idcs=[1]),
            mode=FinalRewMode(state_dependent=True, time_dependent=True)
        )


class QCartPoleSwingUpSim(QCartPoleSim, Serializable):
    """
    Environment in which a pole has to be swung up and stabilized in the upright position (inverted pendulum) by
    moving a cart on a rail. The pole can rotate around an axis perpendicular to direction in which the cart moves.
    """

    name: str = 'qcp-su'

    def __init__(self,
                 dt: float,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None,
                 long: bool = False):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: long (`True`) or short (`False`) pole
        """
        Serializable._init(self, locals())
        super().__init__(dt, max_steps, task_args, long)

    def _create_spaces(self):
        super()._create_spaces()
        l_rail = self.domain_param['l_rail']

        # Define the spaces
        max_state = np.array([+l_rail/2. - self.x_buffer, +4*np.pi, np.inf, np.inf])  # [m, rad, m/s, rad/s]
        min_state = np.array([-l_rail/2. + self.x_buffer, -4*np.pi, -np.inf, -np.inf])  # [m, rad, m/s, rad/s]
        max_init_state = np.array([0.03, 1/180.*np.pi, 0.005, 2/180.*np.pi])  # [m, rad, m/s, rad/s]

        self._state_space = BoxSpace(min_state, max_state, labels=['$x$', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$'])
        self._init_space = BoxSpace(-max_init_state, max_init_state,
                                    labels=['$x$', r'$\theta$', r'$\dot{x}$', r'$\dot{\theta}$'])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', None)
        if state_des is None:
            state_des = np.array([0., np.pi, 0., 0.])

        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, UnderActuatedSwingUpRewFcn(c_act=1e-2), idcs=[1]),
            mode=FinalRewMode(always_negative=True)
        )
