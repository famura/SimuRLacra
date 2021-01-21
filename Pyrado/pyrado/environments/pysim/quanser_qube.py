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

# Added stuff
import math, pathlib

import numpy as np
import torch as to
from abc import abstractmethod
from init_args_serializer.serializable import Serializable

from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.quanser import max_act_qq
from pyrado.spaces.box import BoxSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import RadiallySymmDesStateTask
from pyrado.tasks.reward_functions import ExpQuadrErrRewFcn


class QQubeSim(SimPyEnv, Serializable):
    """ Base Environment for the Quanser Qube swing-up and stabilization task """

    @abstractmethod
    def _create_task(self, task_args: dict) -> Task:
        raise NotImplementedError

    @abstractmethod
    def _create_spaces(self):
        raise NotImplementedError

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            g=9.81,  # gravity [m/s**2]
            Rm=8.4,  # motor resistance [Ohm]
            km=0.042,  # motor back-emf constant [V*s/rad]
            Mr=0.095,  # rotary arm mass [kg]
            Lr=0.085,  # rotary arm length [m]
            Dr=5e-6,  # rotary arm viscous damping [N*m*s/rad], original: 0.0015, identified: 5e-6
            Mp=0.024,  # pendulum link mass [kg]
            Lp=0.129,  # pendulum link length [m]
            Dp=1e-6,
        )  # pendulum link viscous damping [N*m*s/rad], original: 0.0005, identified: 1e-6

    def _calc_constants(self):
        m_r = self.domain_param["Mr"]
        m_p = self.domain_param["Mp"]
        l_r = self.domain_param["Lr"]
        l_p = self.domain_param["Lp"]

        # Moments of inertia
        self._J_r = m_r * l_r ** 2 / 12  # inertia about COM of the rotary pole [kg*m^2]
        self._J_p = m_p * l_p ** 2 / 12  # inertia about COM of the pendulum pole [kg*m^2]
        self._J_p2 = m_p * l_p ** 2 / 4  # Steiner term of the pendulum pole [kg*m^2]
        self._J_pr = m_p * l_p * l_r / 2  # coupled inertia term [kg*m^2]

    def _dyn(self, t, x, u):
        r"""
        Compute $\dot{x} = f(x, u, t)$.

        :param t: time (if the dynamics explicitly depend on the time)
        :param x: state
        :param u: control command
        :return: time derivative of the state
        """
        k_m = self.domain_param["km"]
        R_m = self.domain_param["Rm"]
        d_r = self.domain_param["Dr"]
        d_p = self.domain_param["Dp"]
        m_p = self.domain_param["Mp"]
        l_r = self.domain_param["Lr"]
        l_p = self.domain_param["Lp"]
        g = self.domain_param["g"]

        # Decompose state
        th, al, thd, ald = x
        sin_al = np.sin(al)
        cos_al = np.cos(al)

        # Calculate vector [x, y] = tau - C(q, qd)
        trq_motor = float(k_m * (u - k_m * thd) / R_m)
        trq_rhs_th = self._J_p2 * np.sin(2 * al) * thd * ald - self._J_pr * sin_al * ald ** 2
        trq_rhs_al = -0.5 * self._J_p2 * np.sin(2 * al) * thd ** 2 + 0.5 * m_p * l_p * g * sin_al

        # Compute acceleration from linear system of equations: M * x_ddot = rhs
        M = np.array(
            [
                [self._J_r + m_p * l_r ** 2 + self._J_p2 * sin_al ** 2, self._J_pr * cos_al],
                [self._J_pr * cos_al, self._J_p + 0.25 * m_p * l_p ** 2],
            ]
        )
        rhs = np.array(
            [
                trq_motor - d_r * thd - trq_rhs_th,
                -d_p * ald - trq_rhs_al,
            ]
        )
        thdd, aldd = np.linalg.solve(M, rhs)

        return np.array([thd, ald, thdd, aldd], dtype=np.float64)

    def _step_dynamics(self, act: np.ndarray):
        # Compute the derivative
        thd, ald, thdd, aldd = self._dyn(None, self.state, act)

        # Integration step (Runge-Kutta 4)
        k = np.zeros(shape=(4, 4))  # derivatives
        k[0, :] = np.array([thd, ald, thdd, aldd])
        for j in range(1, 4):
            if j <= 2:
                s = self.state + self._dt / 2.0 * k[j - 1, :]
            else:
                s = self.state + self._dt * k[j - 1, :]
            thd, ald, thdd, aldd = self._dyn(None, self.state, act)
            k[j, :] = np.array([s[2], s[3], thdd, aldd])
        self.state += self._dt / 6 * (k[0] + 2 * k[1] + 2 * k[2] + k[3])

    def _init_anim(self):
        from direct.showbase.ShowBase import ShowBase
        from direct.task import Task
        from panda3d.core import loadPrcFileData, DirectionalLight, AntialiasAttrib, TextNode, WindowProperties, AmbientLight

        # Configuration for panda3d-window
        confVars = """
        win-size 800 600
        window-title Quanser Qube
        framebuffer-multisample 1
        multisamples 2
        """
        loadPrcFileData("", confVars)

        class PandaVis(ShowBase):

        ShowBase.__init__(self)

        mydir = pathlib.Path(__file__).resolve().parent.absolute()

        # Accessing variables of outer class
        self.qq = qq

        # Convert to float for VPython
        Lr = float(self.qq.domain_param["Lr"])
        Lp = float(self.qq.domain_param["Lp"])

        self.setBackgroundColor(1, 1, 1)
        self.cam.setY(-3)
        self.render.setAntialias(AntialiasAttrib.MAuto)
        self.windowProperties = WindowProperties()
        self.windowProperties.setForeground(True)

        # Set lighting
        self.directionalLight = DirectionalLight('directionalLight')
        self.directionalLightNP = self.render.attachNewNode(self.directionalLight)
        self.directionalLightNP.setHpr(0, -8, 0)
        #self.directionalLightNP.setPos(0, 8, 0)
        self.render.setLight(self.directionalLightNP)

        self.ambientLight = AmbientLight('ambientLight')
        self.ambientLight.setColor((0.1, 0.1, 0.1, 1))
        self.ambientLightNP = self.render.attachNewNode(self.ambientLight)
        self.render.setLight(self.ambientLightNP)

        self.text = TextNode('parameters')
        self.textNodePath = aspect2d.attachNewNode(self.text)
        self.textNodePath.setScale(0.07)
        self.textNodePath.setPos(0.3, 0, -0.3)

        # HIER GEHTS WEITER
        scene_range = 0.2
        arm_radius = 0.003
        pole_radius = 0.0045
        self._anim["canvas"].background = vp.color.white
        self._anim["canvas"].lights = []
        vp.distant_light(direction=vp.vec(0.2, 0.2, 0.5), color=vp.color.white)
        self._anim["canvas"].up = vp.vec(0, 0, 1)
        self._anim["canvas"].range = scene_range
        self._anim["canvas"].center = vp.vec(0.04, 0, 0)
        self._anim["canvas"].forward = vp.vec(-2, 1.2, -1)
        vp.box(pos=vp.vec(0, 0, -0.07), length=0.09, width=0.1, height=0.09, color=vp.color.gray(0.5))
        vp.cylinder(axis=vp.vec(0, 0, -1), radius=0.005, length=0.03, color=vp.color.gray(0.5))
        # Joints
        self._anim["joint1"] = vp.sphere(radius=0.005, color=vp.color.white)
        self._anim["joint2"] = vp.sphere(radius=pole_radius, color=vp.color.white)
        # Arm
        self._anim["arm"] = vp.cylinder(radius=arm_radius, length=Lr, color=vp.color.blue)
        # Pole
        self._anim["pole"] = vp.cylinder(radius=pole_radius, length=Lp, color=vp.color.red)
        # Curve
        self._anim["curve"] = vp.curve(color=vp.color.white, radius=0.0005, retain=2000)

    def _update_anim(self):
        self._visualization.taskMgr.step()


        # Convert to float for VPython
        g = self.domain_param["g"]
        Mr = self.domain_param["Mr"]
        Mp = self.domain_param["Mp"]
        Lr = float(self.domain_param["Lr"])
        Lp = float(self.domain_param["Lp"])
        km = self.domain_param["km"]
        Rm = self.domain_param["Rm"]
        Dr = self.domain_param["Dr"]
        Dp = self.domain_param["Dp"]

        th, al, _, _ = self.state
        arm_pos = (Lr * np.cos(th), Lr * np.sin(th), 0.0)
        pole_ax = (-Lp * np.sin(al) * np.sin(th), +Lp * np.sin(al) * np.cos(th), -Lp * np.cos(al))
        self._anim["arm"].axis = vp.vec(*arm_pos)
        self._anim["pole"].pos = vp.vec(*arm_pos)
        self._anim["pole"].axis = vp.vec(*pole_ax)
        self._anim["joint1"].pos = self._anim["arm"].pos
        self._anim["joint2"].pos = self._anim["pole"].pos
        self._anim["curve"].append(self._anim["pole"].pos + self._anim["pole"].axis)

        # Set caption text
        self._anim[
            "canvas"
        ].caption = f"""
            theta: {self.state[0]*180/np.pi : 3.1f}
            alpha: {self.state[1]*180/np.pi : 3.1f}
            dt: {self._dt :1.4f}
            g: {g : 1.3f}
            Mr: {Mr : 1.4f}
            Mp: {Mp : 1.4f}
            Lr: {Lr : 1.4f}
            Lp: {Lp : 1.4f}
            Dr: {Dr : 1.7f}
            Dp: {Dp : 1.7f}
            Rm: {Rm : 1.3f}
            km: {km : 1.4f}
            """

    def _reset_anim(self):
        # Reset VPython animation
        if self._anim["curve"] is not None:
            self._anim["curve"].clear()


class QQubeSwingUpSim(QQubeSim):
    """
    Environment which models Quanser's Furuta pendulum called Quanser Qube.
    The goal is to swing up the pendulum and stabilize at the upright position (alpha = +-pi).
    """

    name: str = "qq-su"

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([115.0 / 180 * np.pi, 4 * np.pi, 20 * np.pi, 20 * np.pi])  # [rad, rad, rad/s, rad/s]
        max_obs = np.array([1.0, 1.0, 1.0, 1.0, np.inf, np.inf])  # [-, -, -, -, rad/s, rad/s]
        max_init_state = np.array([2.0, 1.0, 0.5, 0.5]) / 180 * np.pi  # [rad, rad, rad/s, rad/s]

        self._state_space = BoxSpace(-max_state, max_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])
        self._obs_space = BoxSpace(
            -max_obs, max_obs, labels=["sin_theta", "cos_theta", "sin_alpha", "cos_alpha", "theta_dot", "alpha_dot"]
        )
        self._init_space = BoxSpace(
            -max_init_state, max_init_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"]
        )
        self._act_space = BoxSpace(-max_act_qq, max_act_qq, labels=["V"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([3e-1, 1.0, 2e-2, 5e-3]))
        R = task_args.get("R", np.diag([4e-3]))

        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def observe(self, state, dtype=np.ndarray):
        if dtype is np.ndarray:
            return np.array(
                [np.sin(state[0]), np.cos(state[0]), np.sin(state[1]), np.cos(state[1]), state[2], state[3]]
            )
        elif dtype is to.Tensor:
            return to.cat(
                (
                    state[0].sin().view(1),
                    state[0].cos().view(1),
                    state[1].sin().view(1),
                    state[1].cos().view(1),
                    state[2].view(1),
                    state[3].view(1),
                )
            )


class QQubeStabSim(QQubeSim):
    """
    Environment which models Quanser's Furuta pendulum called Quanser Qube.
    The goal is to stabilize the pendulum at the upright position (alpha = +-pi).

    .. note::
        This environment is only for testing purposes, or to find the PD gains for stabilizing the pendulum at the top.
    """

    name: str = "qq-st"

    def _create_spaces(self):
        # Define the spaces
        max_state = np.array([120.0 / 180 * np.pi, 4 * np.pi, 20 * np.pi, 20 * np.pi])  # [rad, rad, rad/s, rad/s]
        min_init_state = np.array([-5.0 / 180 * np.pi, 175.0 / 180 * np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]
        max_init_state = np.array([5.0 / 180 * np.pi, 185.0 / 180 * np.pi, 0, 0])  # [rad, rad, rad/s, rad/s]

        self._state_space = BoxSpace(-max_state, max_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])
        self._obs_space = self._state_space
        self._init_space = BoxSpace(min_init_state, max_init_state, labels=["theta", "alpha", "theta_dot", "alpha_dot"])
        self._act_space = BoxSpace(-max_act_qq, max_act_qq, labels=["V"])

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.array([0.0, np.pi, 0.0, 0.0]))
        Q = task_args.get("Q", np.diag([3.0, 4.0, 2.0, 2.0]))
        R = task_args.get("R", np.diag([5e-2]))

        return RadiallySymmDesStateTask(self.spec, state_des, ExpQuadrErrRewFcn(Q, R), idcs=[1])

    def observe(self, state, dtype=np.ndarray):
        # Directly observe the noise-free state
        return state.copy()
