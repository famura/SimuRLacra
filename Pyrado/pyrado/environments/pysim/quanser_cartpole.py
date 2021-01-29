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

#added imports
import math, pathlib

import numpy as np
from abc import abstractmethod
from init_args_serializer.serializable import Serializable
from typing import Optional

import pyrado
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.quanser import max_act_qcp
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.final_reward import FinalRewTask, FinalRewMode
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import QuadrErrRewFcn


class QCartPoleSim(SimPyEnv, Serializable):
    """ Base Environment for the Quanser Cart-Pole swing-up and stabilization task """

    def __init__(
        self,
        dt: float,
        max_steps: int,
        task_args: Optional[dict],
        long: bool,
        simple_dynamics: bool,
        wild_init: bool,
    ):
        r"""
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: set to `True` if using the long pole, else `False`
        :param simple_dynamics: if `True, use the simpler dynamics model from Quanser. If `False`, use a dynamics model
                                which includes friction
        :param wild_init: if `True` the init state space is increased drastically, e.g. the initial pendulum angle
                          can be in $[-\pi, +\pi]$. Only applicable to `QCartPoleSwingUpSim`.
        """
        Serializable._init(self, locals())

        self._simple_dynamics = simple_dynamics
        self._th_ddot = None  # internal memory necessary for computing the friction force
        self._obs_space = None
        self._long = long
        self._wild_init = wild_init
        self._x_buffer = 0.05  # [m]

        # Call SimPyEnv's constructor
        super().__init__(dt, max_steps, task_args)

        # Update the class-specific domain parameters
        self.domain_param = self.get_nominal_domain_param(long=long)

    def _create_spaces(self):
        l_rail = self.domain_param["l_rail"]
        max_obs = np.array([l_rail / 2.0, 1.0, 1.0, np.inf, np.inf])

        self._state_space = None
        self._obs_space = BoxSpace(-max_obs, max_obs, labels=["x", "sin_theta", "cos_theta", "x_dot", "theta_dot"])
        self._init_space = None
        self._act_space = BoxSpace(-max_act_qcp, max_act_qcp, labels=["V"])

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        raise NotImplementedError

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Set the initial angular acceleration to zero
        self._th_ddot = 0.0

        return super().reset(init_state, domain_param)

    def observe(self, state) -> np.ndarray:
        return np.array([state[0], np.sin(state[1]), np.cos(state[1]), state[2], state[3]])

    @classmethod
    def get_nominal_domain_param(cls, long: bool = False) -> dict:
        if long:
            m_pole = 0.23
            l_pole = 0.641 / 2
        else:
            m_pole = 0.127
            l_pole = 0.3365 / 2

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
            mu_cart=0.02,  # Coulomb friction coefficient cart-rail [-]
        )

    def _calc_constants(self):
        l_pole = self.domain_param["l_pole"]
        m_pole = self.domain_param["m_pole"]
        m_cart = self.domain_param["m_cart"]
        eta_g = self.domain_param["eta_g"]
        K_g = self.domain_param["K_g"]
        J_m = self.domain_param["J_m"]
        r_mp = self.domain_param["r_mp"]

        self.J_pole = l_pole ** 2 * m_pole / 3.0  # pole inertia [kg*m**2]
        self.J_eq = m_cart + (eta_g * K_g ** 2 * J_m) / r_mp ** 2  # equiv. inertia [kg]

    def _step_dynamics(self, act: np.ndarray):
        g = self.domain_param["g"]
        l_p = self.domain_param["l_pole"]
        m_p = self.domain_param["m_pole"]
        m_c = self.domain_param["m_cart"]
        eta_m = self.domain_param["eta_m"]
        eta_g = self.domain_param["eta_g"]
        K_g = self.domain_param["K_g"]
        R_m = self.domain_param["R_m"]
        k_m = self.domain_param["k_m"]
        r_mp = self.domain_param["r_mp"]
        B_eq = self.domain_param["B_eq"]
        B_p = self.domain_param["B_pole"]
        mu_c = self.domain_param["mu_cart"]

        x, th, x_dot, th_dot = self.state
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        m_tot = m_c + m_p

        # Actuation force coming from the carts motor torque
        f_act = (eta_g * K_g * eta_m * k_m) / (R_m * r_mp) * (eta_m * act - K_g * k_m * x_dot / r_mp)

        if self._simple_dynamics:
            f_tot = float(f_act)

        else:
            # Force normal to the rail causing the Coulomb friction
            f_normal = m_tot * g - m_p * l_p / 2 * (sin_th * self._th_ddot + cos_th * th_dot ** 2)
            if f_normal < 0:
                # The normal force on the cart is negative, i.e. it is lifted up. This can be cause by a very high
                # angular momentum of the pole
                f_c = 0.0
            else:
                f_c = mu_c * f_normal * np.sign(f_normal * x_dot)
            f_tot = float(f_act - f_c)

        M = np.array(
            [
                [m_p + self.J_eq, m_p * l_p * cos_th],
                [m_p * l_p * cos_th, self.J_pole + m_p * l_p ** 2],
            ]
        )
        rhs = np.array(
            [
                f_tot - B_eq * x_dot - m_p * l_p * sin_th * th_dot ** 2,
                -B_p * th_dot - m_p * l_p * g * sin_th,
            ]
        )
        # Compute acceleration from linear system of equations: M * x_ddot = rhs
        x_ddot, self._th_ddot = np.linalg.solve(M, rhs)

        # Integration step (symplectic Euler)
        self.state[2:] += np.array([float(x_ddot), float(self._th_ddot)]) * self._dt  # next velocity
        self.state[:2] += self.state[2:] * self._dt  # next position

    def _init_anim(self):
        import vpython as vp
        from direct.showbase.ShowBase import ShowBase
        from direct.task import Task
        from panda3d.core import loadPrcFileData, DirectionalLight, AntialiasAttrib, TextNode, AmbientLight, WindowProperties

        # Configuration for panda3d-window
        confVars = """
               win-size 800 600
               window-title Ball on Beam
               framebuffer-multisample 1
               multisamples 2
               """
        loadPrcFileData("", confVars)

        class PandaVis(ShowBase):
            def __init__(self, qcp):
                ShowBase.__init__(self)

                mydir = pathlib.Path(__file__).resolve().parent.absolute()

                self.setBackgroundColor(0, 0, 0, 1)
                
                #Accessing variables of outer class
                self.qcp = qcp

                #setting parameters
                l_pole = float(self.qcp.domain_param["l_pole"])
                l_rail = float(self.qcp.domain_param["l_rail"])

                # Only for animation
                l_cart, h_cart = 0.08, 0.08
                r_pole, r_rail = 0.01, 0.005

                # Get positions
                x, th, _, _ = self.qcp.state

                self.setBackgroundColor(0, 0, 0)
                self.cam.setY(-5)
                self.render.setAntialias(AntialiasAttrib.MAuto)
                self.windowProperties = WindowProperties()
                self.windowProperties.setForeground(True)

                # Set lighting
                self.directionalLight = DirectionalLight('directionalLight')
                self.directionalLightNP = self.render.attachNewNode(self.directionalLight)
                self.directionalLightNP.setHpr(0, -8, 0)
                # self.directionalLightNP.setPos(0, 8, 0)
                self.render.setLight(self.directionalLightNP)

                self.ambientLight = AmbientLight('ambientLight')
                self.ambientLight.setColor((0.1, 0.1, 0.1, 1))
                self.ambientLightNP = self.render.attachNewNode(self.ambientLight)
                self.render.setLight(self.ambientLightNP)

                # Text
                self.text = TextNode('parameters')
                self.textNodePath = aspect2d.attachNewNode(self.text)
                self.textNodePath.setScale(0.07)
                self.textNodePath.setPos(0.3, 0, -0.3)

                #TODO RenderEinstellungen pro Objekt

                #Rail
                self.rail = self.loader.loadModel(pathlib.Path(mydir, "cylinder_shifted_center.egg"))
                self.rail.setPos(-l_rail/2, 0, -h_cart/2 - r_rail)
                self.rail.setScale(r_rail, r_rail, l_rail)
                self.rail.setColor(1, 1, 1) #white
                self.rail.reparentTo(self.render)
                self.rail.setR(90)

                #Cart
                self.cart = self.loader.loadModel(pathlib.Path(mydir, "box.egg"))
                self.cart.setPos(x, 0, 0)
                self.cart.setScale(l_cart, h_cart/2, h_cart)
                self.cart.setColor(0, 1, 0) #green
                self.cart.reparentTo(self.render)

                #Joint
                self.joint = self.loader.loadModel(pathlib.Path(mydir, "ball.egg"))
                self.joint.setPos(x, r_pole + h_cart/4, 0)
                self.joint.setScale(r_pole, r_pole, r_pole)
                self.joint.setColor(0, 0, 0) #white
                self.joint.reparentTo(self.render)

                #Pole
                self.pole = self.loader.loadModel(pathlib.Path(mydir, "cylinder_shifted_center.egg"))
                self.pole.setPos(x, r_pole + h_cart/4, -l_pole)
                self.pole.setHpr(2 * l_pole * np.sin(th), -2 * l_pole * np.cos(th), 0)
                #H um Z-Achse, P um X-Achse, R um Y-Achse
                self.pole.setScale(r_pole, r_pole, 2*l_pole)
                self.pole.setColor(0, 0, 1) #blue
                self.pole.reparentTo(self.render)

                #Update-Aufruf
                self.taskMgr.add(self.update, "update")

            def update(self, task):

                g = self.qcp.domain_param["g"]
                m_cart = self.qcp.domain_param["m_cart"]
                m_pole = self.qcp.domain_param["m_pole"]
                l_pole = float(self.qcp.domain_param["l_pole"])
                l_rail = float(self.qcp.domain_param["l_rail"])
                eta_m = self.qcp.domain_param["eta_m"]
                eta_g = self.qcp.domain_param["eta_g"]
                K_g = self.qcp.domain_param["K_g"]
                J_m = self.qcp.domain_param["J_m"]
                R_m = self.qcp.domain_param["R_m"]
                k_m = self.qcp.domain_param["k_m"]
                r_mp = self.qcp.domain_param["r_mp"]
                B_eq = self.qcp.domain_param["B_eq"]
                B_pole = self.qcp.domain_param["B_pole"]

                # Only for animation
                l_cart, h_cart = 0.08, 0.08
                r_pole, r_rail = 0.01, 0.005

                # Get positions
                x, th, _, _ = self.qcp.state

                # Rail sollte doch eigentlich fest bleiben?!
                self.rail.setX(-l_rail / 2)
                self.rail.setSz(l_rail)

                #Cart
                #self._anim["cart"].pos = vp.vec(x, 0, 0)
                self.cart.setX(x)

                #Joint
                #self._anim["joint"].pos = vp.vec(x, 0, r_pole + h_cart / 4)
                self.joint.setX(x)

                #Pole
                #self._anim["pole"].pos = vp.vec(x, 0, r_pole + h_cart / 4)
                self.pole.setX(x)
                #Original
                #self._anim["pole"].axis = vp.vec(2 * l_pole * vp.sin(th), -2 * l_pole * vp.cos(th), 0)

                #Übersetzung 1:1
                #self.pole.setHpr(2 * l_pole * np.sin(th) *180/np.pi, 0, -2 * l_pole * vp.cos(th) *180/np.pi)

                self.pole.setR(-th * 180/np.pi )

                #Vertauschte P-R-Achsen
                #self.pole.setHpr(2 * l_pole * np.sin(th) *180/np.pi, -2 * l_pole * np.cos(th) *180/np.pi, 0)
                print(th)

                self.text.setText(f"""
                                    th: {th}
                                    x: {self.qcp.state[0] : 1.4f}
                                    theta: {self.qcp.state[1]*180/np.pi : 2.3f}
                                    dt: {self.qcp._dt :1.4f}
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
                                    """)
                self.text.setTextColor(1, 1, 1, 1) #white
                return Task.cont

            def reset(self):
                pass

        # Create instance of PandaVis
        self._visualization = PandaVis(self)
        # States that visualization is running
        self._initiated = True

    def _update_anim(self):
        self._visualization.taskMgr.step()
        #import vpython as vp


        '''
        # Cart
        self._anim["cart"].pos = vp.vec(x, 0, 0)
        self._anim["joint"].pos = vp.vec(x, 0, r_pole + h_cart / 4)'''
        '''
        # Pole
        self._anim["pole"].pos = vp.vec(x, 0, r_pole + h_cart / 4)
        self._anim["pole"].axis = vp.vec(2 * l_pole * vp.sin(th), -2 * l_pole * vp.cos(th), 0)

        # Set caption text
        self._anim[
            "canvas"
        ].caption = f"""
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
                    """'''
        def _reset_anim(self):
            self._visualization.reset()


class QCartPoleStabSim(QCartPoleSim, Serializable):
    """
    Environment in which a pole has to be stabilized in the upright position (inverted pendulum) by moving a cart on
    a rail. The pole can rotate around an axis perpendicular to direction in which the cart moves.
    """

    name: str = "qcp-st"

    def __init__(
        self,
        dt: float,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        long: bool = True,
        simple_dynamics: bool = True,
    ):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: set to `True` if using the long pole, else `False`
        :param simple_dynamics: if `True, use the simpler dynamics model from Quanser. If `False`, use a dynamics model
                                which includes friction
        """
        Serializable._init(self, locals())

        self.stab_thold = 15 / 180.0 * np.pi  # threshold angle for the stabilization task to be a failure [rad]
        self.max_init_th_offset = 8 / 180.0 * np.pi  # [rad]

        super().__init__(dt, max_steps, task_args, long, simple_dynamics, wild_init=False)

    def _create_spaces(self):
        super()._create_spaces()
        l_rail = self.domain_param["l_rail"]

        min_state = np.array(
            [-l_rail / 2.0 + self._x_buffer, np.pi - self.stab_thold, -l_rail, -2 * np.pi]
        )  # [m, rad, m/s, rad/s]
        max_state = np.array(
            [+l_rail / 2.0 - self._x_buffer, np.pi + self.stab_thold, +l_rail, +2 * np.pi]
        )  # [m, rad, m/s, rad/s]

        max_init_state = np.array(
            [+0.02, np.pi + self.max_init_th_offset, +0.02, +5 / 180 * np.pi]
        )  # [m, rad, m/s, rad/s]
        min_init_state = np.array(
            [-0.02, np.pi - self.max_init_th_offset, -0.02, -5 / 180 * np.pi]
        )  # [m, rad, m/s, rad/s]

        self._state_space = BoxSpace(min_state, max_state, labels=["x", "theta", "x_dot", "theta_dot"])
        self._init_space = BoxSpace(min_init_state, max_init_state, labels=["x", "theta", "x_dot", "theta_dot"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([5e-0, 1e1, 1e-2, 1e-2]))
        R = task_args.get("R", np.diag([1e-3]))

        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, QuadrErrRewFcn(Q, R), idcs=[1]),
            mode=FinalRewMode(state_dependent=True, time_dependent=True),
        )


class QCartPoleSwingUpSim(QCartPoleSim, Serializable):
    """
    Environment in which a pole has to be swung up and stabilized in the upright position (inverted pendulum) by
    moving a cart on a rail. The pole can rotate around an axis perpendicular to direction in which the cart moves.
    """

    name: str = "qcp-su"

    def __init__(
        self,
        dt: float,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        long: bool = False,
        simple_dynamics: bool = False,
        wild_init: bool = True,
    ):
        r"""
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param long: set to `True` if using the long pole, else `False`
        :param simple_dynamics: if `True, use the simpler dynamics model from Quanser. If `False`, use a dynamics model
                                which includes friction
        :param wild_init: if `True` the init state space is increased drastically, e.g. the initial pendulum angle
                          can be in $[-\pi, +\pi]$
        """
        Serializable._init(self, locals())

        super().__init__(dt, max_steps, task_args, long, simple_dynamics, wild_init)

    def _create_spaces(self):
        super()._create_spaces()

        # Define the spaces
        l_rail = self.domain_param["l_rail"]
        max_state = np.array(
            [+l_rail / 2.0 - self._x_buffer, +4 * np.pi, 2 * l_rail, 20 * np.pi]
        )  # [m, rad, m/s, rad/s]
        min_state = np.array(
            [-l_rail / 2.0 + self._x_buffer, -4 * np.pi, -2 * l_rail, -20 * np.pi]
        )  # [m, rad, m/s, rad/s]
        if self._wild_init:
            max_init_state = np.array([0.05, np.pi, 0.02, 2 / 180.0 * np.pi])  # [m, rad, m/s, rad/s]
        else:
            max_init_state = np.array([0.02, 2 / 180.0 * np.pi, 0.0, 1 / 180.0 * np.pi])  # [m, rad, m/s, rad/s]

        self._state_space = BoxSpace(min_state, max_state, labels=["x", "theta", "x_dot", "theta_dot"])
        self._init_space = BoxSpace(-max_init_state, max_init_state, labels=["x", "theta", "x_dot", "theta_dot"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([3e-1, 5e-1, 5e-3, 1e-3]))
        R = task_args.get("R", np.diag([1e-3]))
        rew_fcn = QuadrErrRewFcn(Q, R)

        return FinalRewTask(
            RadiallySymmDesStateTask(self.spec, state_des, rew_fcn, idcs=[1]),
            mode=FinalRewMode(always_negative=True),
            factor=1e4,
        )
