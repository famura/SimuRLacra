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

import os
import os.path as osp
from typing import Optional

import numpy as np
import torch as to
from init_args_serializer.serializable import Serializable

import pyrado
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.quanser import MAX_ACT_QBB
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.polar import Polar2DPosVelSpace
from pyrado.tasks.base import Task
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import ScaledExpQuadrErrRewFcn
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt_once


class QBallBalancerSim(SimPyEnv, Serializable):
    """
    Environment in which a ball rolls on an actuated plate. The ball is randomly initialized on the plate and is to be
    stabilized on the center of the plate. The problem formulation treats this setup as 2 independent ball-on-beam
    problems. The plate is actuated via 2 servo motors that lift the plate.

    .. note::
        The dynamics are not the same as in the Quanser Workbook (2 DoF Ball-Balancer - Instructor). Here, we added
        the coriolis forces and linear-viscous friction. However, the 2 dim system is still modeled to be decoupled.
        This is the case, since the two rods (connected to the servos) are pushing the plate at the center lines.
        As a result, the angles alpha and beta are w.r.t. to the inertial frame, i.e. they are not 2 sequential rations.
    """

    name: str = "qbb"

    def __init__(
        self,
        dt: float,
        max_steps: int = pyrado.inf,
        task_args: Optional[dict] = None,
        simple_dynamics: bool = False,
        load_experimental_tholds: bool = True,
    ):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param simple_dynamics: if `True, use a dynamics model without Coriolis forces and without friction effects
        :param load_experimental_tholds: use the voltage thresholds determined from experiments
        """
        Serializable._init(self, locals())

        self._simple_dynamics = simple_dynamics
        self.plate_angs = np.zeros(2)  # plate's angles alpha and beta [rad] (unused for simple_dynamics = True)

        # Call SimPyEnv's constructor
        super().__init__(dt, max_steps, task_args)

        if not simple_dynamics:
            self._kin = QBallBalancerKin(self)

    def _create_spaces(self):
        l_plate = self.domain_param["plate_length"]

        # Define the spaces
        max_state = np.array(
            [
                np.pi / 4.0,
                np.pi / 4.0,
                l_plate / 2.0,
                l_plate / 2.0,  # [rad, rad, m, m, ...
                5 * np.pi,
                5 * np.pi,
                0.5,
                0.5,
            ]
        )  # ... rad/s, rad/s, m/s, m/s]
        min_init_state = np.array([0.75 * l_plate / 2, -np.pi, -0.05 * max_state[6], -0.05 * max_state[7]])
        max_init_state = np.array([0.8 * l_plate / 2, np.pi, 0.05 * max_state[6], 0.05 * max_state[7]])

        self._state_space = BoxSpace(
            -max_state,
            max_state,
            labels=["theta_x", "theta_y", "x", "y", "theta_x_dot", "theta_y_dot", "x_dot", "y_dot"],
        )
        self._obs_space = self._state_space.copy()
        self._init_space = Polar2DPosVelSpace(min_init_state, max_init_state, labels=["r", "phi", "x_dot", "y_dot"])
        self._act_space = BoxSpace(-MAX_ACT_QBB, MAX_ACT_QBB, labels=["V_x", "V_y"])

        self._curr_act = np.zeros_like(MAX_ACT_QBB)  # just for usage in render function

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

    # Cache measured thresholds during one run and reduce console log spam that way
    measured_tholds = None

    @classmethod
    def get_voltage_tholds(cls, load_experiments: bool = True) -> dict:
        """If available, the voltage thresholds computed from measurements, else use default values."""
        # Hard-coded default thresholds
        tholds = dict(
            voltage_thold_x_pos=0.28, voltage_thold_x_neg=-0.10, voltage_thold_y_pos=0.28, voltage_thold_y_neg=-0.074
        )

        if load_experiments:
            if cls.measured_tholds is None:
                ex_dir = osp.join(pyrado.EVAL_DIR, "volt_thold_qbb")
                if osp.exists(ex_dir) and osp.isdir(ex_dir) and os.listdir(ex_dir):
                    print_cbt_once("Found measured thresholds, using the averages.", "g")
                    # Calculate cumulative running average
                    cma = np.zeros((2, 2))
                    i = 0.0
                    for f in filter(lambda f: f.endswith(".npy"), os.listdir(".npy")):
                        i += 1.0
                        cma = cma + (np.load(osp.join(ex_dir, f)) - cma) / i
                    tholds["voltage_thold_x_pos"] = cma[0, 1]
                    tholds["voltage_thold_x_neg"] = cma[0, 0]
                    tholds["voltage_thold_y_pos"] = cma[1, 1]
                    tholds["voltage_thold_y_neg"] = cma[1, 0]
                else:
                    print_cbt_once("No measured thresholds found, falling back to default values.", "y")

                # Cache results for future calls
                cls.measured_tholds = tholds
            else:
                tholds = cls.measured_tholds

        return tholds

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        voltage_tholds = cls.get_voltage_tholds()
        return dict(
            gravity_const=9.81,  # gravity constant [m/s**2]
            ball_mass=0.003,  # mass of the ball [kg]
            ball_radius=0.019625,  # radius of the ball [m]
            plate_length=0.275,  # length of the (square) plate [m]
            arm_radius=0.0254,  # distance between the servo output gear shaft and the coupled joint [m]
            gear_ratio=70.0,  # gear ratio [-]
            gear_efficiency=0.9,  # gearbox efficiency [-]
            load_inertia=5.2822e-5,  # load moment of inertia [kg*m**2]
            motor_inertia=4.6063e-7,  # motor moment of inertia [kg*m**2]
            motor_back_emf=0.0077,  # motor torque constant [N*m/A] = back-EMF constant [V*s/rad]
            motor_resistance=2.6,  # motor armature resistance
            motor_efficiency=0.69,  # motor efficiency [-]
            combined_damping=0.015,  # equivalent viscous damping coefficient w.r.t. load [N*m*s/rad]
            ball_damping=0.05,  # viscous damping coefficient for the ball velocity [N*s/m]
            voltage_thold_x_pos=voltage_tholds[
                "voltage_thold_x_pos"
            ],  # min. voltage required to move the x servo in pos. dir. [V]
            voltage_thold_x_neg=voltage_tholds[
                "voltage_thold_x_neg"
            ],  # min. voltage required to move the x servo in neg. dir. [V]
            voltage_thold_y_pos=voltage_tholds[
                "voltage_thold_y_pos"
            ],  # min. voltage required to move the y servo in pos. dir. [V]
            voltage_thold_y_neg=voltage_tholds[
                "voltage_thold_y_neg"
            ],  # min. voltage required to move the y servo in neg. dir. [V]
            offset_th_x=0.0,  # angular offset of the x axis motor shaft [rad]
            offset_th_y=0.0,  # angular offset of the y axis motor shaft [rad]
        )

    def _calc_constants(self):
        l_plate = self.domain_param["plate_length"]
        m_ball = self.domain_param["ball_mass"]
        r_ball = self.domain_param["ball_radius"]
        eta_g = self.domain_param["gear_efficiency"]
        eta_m = self.domain_param["motor_efficiency"]
        K_g = self.domain_param["gear_ratio"]
        J_m = self.domain_param["motor_inertia"]
        J_l = self.domain_param["load_inertia"]
        r_arm = self.domain_param["arm_radius"]
        k_m = self.domain_param["motor_back_emf"]
        R_m = self.domain_param["motor_resistance"]
        B_eq = self.domain_param["combined_damping"]

        self.J_ball = 2.0 / 5 * m_ball * r_ball ** 2  # inertia of the ball [kg*m**2]
        self.J_eq = eta_g * K_g ** 2 * J_m + J_l  # equivalent moment of inertia [kg*m**2]
        self.c_kin = 2.0 * r_arm / l_plate  # coefficient for the rod-plate kinematic
        self.A_m = eta_g * K_g * eta_m * k_m / R_m
        self.B_eq_v = eta_g * K_g ** 2 * eta_m * k_m ** 2 / R_m + B_eq
        self.zeta = m_ball * r_ball ** 2 + self.J_ball  # combined moment of inertial for the ball

    def _state_from_init(self, init_state):
        state = np.zeros(8)
        state[2:4] = init_state[:2]
        state[6:8] = init_state[2:]
        return state

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        obs = super().reset(init_state=init_state, domain_param=domain_param)

        # Reset the plate angles
        if self._simple_dynamics:
            self.plate_angs = np.zeros(2)  # actually not necessary since not used
        else:
            offset_th_x = self.domain_param["offset_th_x"]
            offset_th_y = self.domain_param["offset_th_y"]
            # Get the plate angles from inverse kinematics for initial pose
            self.plate_angs[0] = self._kin(self.state[0] + offset_th_x)
            self.plate_angs[1] = self._kin(self.state[1] + offset_th_y)

        # Return perfect observation
        return obs

    def _step_dynamics(self, act: np.ndarray):
        gravity_const = self.domain_param["gravity_const"]
        m_ball = self.domain_param["ball_mass"]
        r_ball = self.domain_param["ball_radius"]
        ball_damping = self.domain_param["ball_damping"]
        V_thold_x_neg = self.domain_param["voltage_thold_x_neg"]
        V_thold_x_pos = self.domain_param["voltage_thold_x_pos"]
        V_thold_y_neg = self.domain_param["voltage_thold_y_neg"]
        V_thold_y_pos = self.domain_param["voltage_thold_y_pos"]
        offset_th_x = self.domain_param["offset_th_x"]
        offset_th_y = self.domain_param["offset_th_y"]

        # Apply a voltage dead zone, i.e. below a certain amplitude the system is will not move. This is a very
        # simple model of static friction. Experimentally evaluated the voltage required to get the plate moving.
        if not self._simple_dynamics and V_thold_x_neg <= act[0] <= V_thold_x_pos:
            act[0] = 0
        if not self._simple_dynamics and V_thold_y_neg <= act[1] <= V_thold_y_pos:
            act[1] = 0

        # State
        th_x = self.state[0] + offset_th_x  # angle of the x axis servo (load)
        th_y = self.state[1] + offset_th_y  # angle of the y axis servo (load)
        x = self.state[2]  # ball position along the x axis
        y = self.state[3]  # ball position along the y axis
        th_x_dot = self.state[4]  # angular velocity of the x axis servo (load)
        th_y_dot = self.state[5]  # angular velocity of the y axis servo (load)
        x_dot = self.state[6]  # ball velocity along the x axis
        y_dot = self.state[7]  # ball velocity along the y axis

        th_x_ddot = (self.A_m * act[0] - self.B_eq_v * th_x_dot) / self.J_eq
        th_y_ddot = (self.A_m * act[1] - self.B_eq_v * th_y_dot) / self.J_eq

        """
        THIS IS TIME INTENSIVE
        if not self._simple_dynamics:
            # Get the plate angles from inverse kinematics
            self.plate_angs[0] = self._kin(self.state[0] + self.offset_th_x)
            self.plate_angs[1] = self._kin(self.state[1] + self.offset_th_y)
        """

        # Plate (not part of the state since it is a redundant information)
        # The definition of th_y is opposing beta, i.e.
        a = self.plate_angs[0]  # plate's angle around the y axis (alpha)
        b = self.plate_angs[1]  # plate's angle around the x axis (beta)
        a_dot = self.c_kin * th_x_dot * np.cos(th_x) / np.cos(a)  # plate's angular velocity around the y axis (alpha)
        b_dot = self.c_kin * -th_y_dot * np.cos(-th_y) / np.cos(b)  # plate's angular velocity around the x axis (beta)
        # Plate's angular accelerations (unused for simple_dynamics = True)
        a_ddot = (
            1.0
            / np.cos(a)
            * (self.c_kin * (th_x_ddot * np.cos(th_x) - th_x_dot ** 2 * np.sin(th_x)) + a_dot ** 2 * np.sin(a))
        )
        b_ddot = (
            1.0
            / np.cos(b)
            * (self.c_kin * (-th_y_ddot * np.cos(th_y) - (-th_y_dot) ** 2 * np.sin(-th_y)) + b_dot ** 2 * np.sin(b))
        )

        # kinematics: sin(a) = self.c_kin * sin(th_x)
        if self._simple_dynamics:
            # Ball dynamic without friction and Coriolis forces
            x_ddot = self.c_kin * m_ball * gravity_const * r_ball ** 2 * np.sin(th_x) / self.zeta  # symm inertia
            y_ddot = self.c_kin * m_ball * gravity_const * r_ball ** 2 * np.sin(th_y) / self.zeta  # symm inertia
        else:
            # Ball dynamic with friction and Coriolis forces
            x_ddot = (
                -ball_damping * x_dot * r_ball ** 2  # friction
                - self.J_ball * r_ball * a_ddot  # plate influence
                + m_ball * x * a_dot ** 2 * r_ball ** 2  # centripetal
                + self.c_kin * m_ball * gravity_const * r_ball ** 2 * np.sin(th_x)  # gravity
            ) / self.zeta
            y_ddot = (
                -ball_damping * y_dot * r_ball ** 2  # friction
                - self.J_ball * r_ball * b_ddot  # plate influence
                + m_ball * y * (-b_dot) ** 2 * r_ball ** 2  # centripetal
                + self.c_kin * m_ball * gravity_const * r_ball ** 2 * np.sin(th_y)  # gravity
            ) / self.zeta

        # Integration step (symplectic Euler)
        self.state[4:] += np.array([th_x_ddot, th_y_ddot, x_ddot, y_ddot]) * self._dt  # next velocity
        self.state[:4] += self.state[4:] * self._dt  # next position

        # Integration step (forward Euler)
        self.plate_angs += np.array([a_dot, b_dot]) * self._dt  # just for debugging when simplified dynamics

    def _init_anim(self):
        # Import PandaVis Class
        from pyrado.environments.pysim.pandavis import QBallBalancerVis

        # Create instance of PandaVis
        self._visualization = QBallBalancerVis(self, self._rendering)


class QBallBalancerKin(Serializable):
    """
    Calculates and visualizes the kinematics from the servo shaft angles (th_x, th_x) to the plate angles (a, b).
    """

    def __init__(self, qbb, num_opt_iter=100, render_mode=RenderMode()):
        """
        Constructor

        :param qbb: QBallBalancerSim object
        :param num_opt_iter: number of optimizer iterations for the IK
        :param mode: the render mode: a for animating (pyplot), or `` for no animation
        """
        from matplotlib import pyplot as plt

        Serializable._init(self, locals())

        self._qbb = qbb
        self.num_opt_iter = num_opt_iter
        self.render_mode = render_mode

        self.r = float(self._qbb.domain_param["arm_radius"])
        self.l = float(self._qbb.domain_param["plate_length"] / 2.0)
        self.d = 0.10  # [m] roughly measured

        # Visualization
        if render_mode.video:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.ax.set_xlim(-0.5 * self.r, 1.2 * (self.r + self.l))
            self.ax.set_ylim(-1.0 * self.d, 2 * self.d)
            self.ax.set_aspect("equal")
            (self.line1,) = self.ax.plot([0, 0], [0, 0], marker="o")
            (self.line2,) = self.ax.plot([0, 0], [0, 0], marker="o")
            (self.line3,) = self.ax.plot([0, 0], [0, 0], marker="o")

    def __call__(self, th):
        """
        Compute the inverse kinematics of the Quanser 2 DoF Ball-Balancer for one DoF

        :param th: angle of the servo (x or y axis)
        :return: plate angle al pha or beta
        """
        from matplotlib import pyplot as plt

        if not isinstance(th, to.Tensor):
            th = to.tensor(th, dtype=to.get_default_dtype(), requires_grad=False)

        # Update the lengths, e.g. if the domain has been randomized
        # Need to use float() since the parameters might be 0d-arrays
        self.r = float(self._qbb.domain_param["arm_radius"])
        self.l = float(self._qbb.domain_param["plate_length"] / 2.0)
        self.d = 0.10  # roughly measured

        tip = self.rod_tip(th)
        ang = self.plate_ang(tip)

        if self.render_mode.video:
            self.render(th, tip)
            plt.pause(0.001)

        return ang

    @to.enable_grad()
    def rod_tip(self, th):
        """
        Get Cartesian coordinates of the rod tip for one servo.

        :param th: current value of the respective servo shaft angle
        :return tip: 2D position of the rod tip in the sagittal plane
        """
        # Initial guess for the rod tip
        tip_init = [self.r, self.l]  # [x, y] in the sagittal plane
        tip = to.tensor(tip_init, requires_grad=True)

        optim = to.optim.SGD([tip], lr=0.01, momentum=0.9)
        for i in range(self.num_opt_iter):
            optim.zero_grad()
            loss = self._loss_fcn(tip, th)
            loss.backward()
            optim.step()

        return tip

    def _loss_fcn(self, tip, th):
        """
        Cost function for the optimization problem, which only consists of 2 constraints that should be fulfilled.

        :param tip:
        :param th:
        :return: the cost value
        """
        # Formulate the constrained optimization problem as an unconstrained using the known segment lengths
        rod_len = to.sqrt((tip[0] - self.r * to.cos(th)) ** 2 + (tip[1] - self.r * to.sin(th)) ** 2)
        half_palte = to.sqrt((tip[0] - self.r - self.l) ** 2 + (tip[1] - self.d) ** 2)

        return (rod_len - self.d) ** 2 + (half_palte - self.l) ** 2

    def plate_ang(self, tip):
        """
        Compute plate angle (alpha or beta) from the rod tip position which has been calculated from servo shaft angle
        (th_x or th_y) before.
        :return tip: 2D position of the rod tip in the sagittal plane (from the optimizer)
        """
        ang = np.pi / 2.0 - to.atan2(self.r + self.l - tip[0], tip[1] - self.d)
        return float(ang)

    def render(self, th, tip):
        """
        Visualize using pyplot

        :param th: angle of the servo
        :param tip: 2D position of the rod tip in the sagittal plane (from the optimizer)
        """
        A = [0, 0]
        B = [self.r * np.cos(th), self.r * np.sin(th)]
        C = [tip[0], tip[1]]
        D = [self.r + self.l, self.d]

        self.line1.set_data([A[0], B[0]], [A[1], B[1]])
        self.line2.set_data([B[0], C[0]], [B[1], C[1]])
        self.line3.set_data([C[0], D[0]], [C[1], D[1]])

    def _get_state(self, state_dict):
        state_dict["r"] = self.r
        state_dict["l"] = self.l

    def _set_state(self, state_dict, copying=False):
        self.r = state_dict["r"]
        self.l = state_dict["l"]
