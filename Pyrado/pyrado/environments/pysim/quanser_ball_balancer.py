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
import numpy as np
import torch as to
from init_args_serializer.serializable import Serializable

import pyrado
from matplotlib import pyplot as plt
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.quanser import max_act_qbb
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

    name: str = 'qbb'

    def __init__(self,
                 dt: float,
                 max_steps: int = pyrado.inf,
                 task_args: [dict, None] = None,
                 simplified_dyn: bool = False,
                 load_experimental_tholds: bool = True):
        """
        Constructor

        :param dt: simulation step size [s]
        :param max_steps: maximum number of simulation steps
        :param task_args: arguments for the task construction
        :param simplified_dyn: use a dynamics model without Coriolis forces and without friction
        :param load_experimental_tholds: use the voltage thresholds determined from experiments
        """
        Serializable._init(self, locals())

        self._simplified_dyn = simplified_dyn
        self.plate_angs = np.zeros(2)  # plate's angles alpha and beta [rad] (unused for simplified_dyn = True)

        # Call SimPyEnv's constructor
        super().__init__(dt, max_steps, task_args)

        if not simplified_dyn:
            self._kin = QBallBalancerKin(self)

    def _create_spaces(self):
        l_plate = self.domain_param['l_plate']

        # Define the spaces
        max_state = np.array([np.pi/4., np.pi/4., l_plate/2., l_plate/2.,  # [rad, rad, m, m, ...
                              5*np.pi, 5*np.pi, 0.5, 0.5])  # ... rad/s, rad/s, m/s, m/s]
        min_init_state = np.array([0.75*l_plate/2, -np.pi, -0.05*max_state[6], -0.05*max_state[7]])
        max_init_state = np.array([0.8*l_plate/2, np.pi, 0.05*max_state[6], 0.05*max_state[7]])

        self._state_space = BoxSpace(-max_state, max_state, labels=['theta_x', 'theta_y', 'x', 'y',
                                                                    'theta_x_dot', 'theta_y_dot', 'x_dot', 'y_dot'])
        self._obs_space = self._state_space.copy()
        self._init_space = Polar2DPosVelSpace(min_init_state, max_init_state, labels=['r', 'phi', 'x_dot', 'y_dot'])
        self._act_space = BoxSpace(-max_act_qbb, max_act_qbb, labels=['V_x', 'V_y'])

        self._curr_act = np.zeros_like(max_act_qbb)  # just for usage in render function

    def _create_task(self, task_args: dict) -> Task:
        # Define the task including the reward function
        state_des = task_args.get('state_des', np.zeros(8))
        Q = task_args.get('Q', np.diag([1e0, 1e0, 5e3, 5e3, 1e-2, 1e-2, 5e-1, 5e-1]))
        R = task_args.get('R', np.diag([1e-2, 1e-2]))
        # Q = np.diag([1e2, 1e2, 5e2, 5e2, 1e-2, 1e-2, 1e+1, 1e+1])  # for LQR
        # R = np.diag([1e-2, 1e-2])  # for LQR

        return DesStateTask(
            self.spec, state_des, ScaledExpQuadrErrRewFcn(Q, R, self.state_space, self.act_space, min_rew=1e-4)
        )

    # Cache measured thresholds during one run and reduce console log spam that way
    measured_tholds = None

    @classmethod
    def get_V_tholds(cls, load_experiments: bool = True) -> dict:
        """ If available, the voltage thresholds computed from measurements, else use default values. """
        # Hard-coded default thresholds
        tholds = dict(V_thold_x_pos=0.28,
                      V_thold_x_neg=-0.10,
                      V_thold_y_pos=0.28,
                      V_thold_y_neg=-0.074)

        if load_experiments:
            if cls.measured_tholds is None:
                ex_dir = osp.join(pyrado.EVAL_DIR, 'volt_thold_qbb')
                if osp.exists(ex_dir) and osp.isdir(ex_dir) and os.listdir(ex_dir):
                    print_cbt_once('Found measured thresholds, using the averages.', 'g')
                    # Calculate cumulative running average
                    cma = np.zeros((2, 2))
                    i = 0.
                    for f in os.listdir(ex_dir):
                        if f.endswith('.npy'):
                            i += 1.
                            cma = cma + (np.load(osp.join(ex_dir, f)) - cma)/i
                    tholds['V_thold_x_pos'] = cma[0, 1]
                    tholds['V_thold_x_neg'] = cma[0, 0]
                    tholds['V_thold_y_pos'] = cma[1, 1]
                    tholds['V_thold_y_neg'] = cma[1, 0]
                else:
                    print_cbt_once('No measured thresholds found, falling back to default values.', 'y')

                # Cache results for future calls
                cls.measured_tholds = tholds
            else:
                tholds = cls.measured_tholds

        return tholds

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        V_tholds = cls.get_V_tholds()
        return dict(g=9.81,  # gravity constant [m/s**2]
                    m_ball=0.003,  # mass of the ball [kg]
                    r_ball=0.019625,  # radius of the ball [m]
                    l_plate=0.275,  # length of the (square) plate [m]
                    r_arm=0.0254,  # distance between the servo output gear shaft and the coupled joint [m]
                    K_g=70.,  # gear ratio [-]
                    eta_g=0.9,  # gearbox efficiency [-]
                    J_l=5.2822e-5,  # load moment of inertia [kg*m**2]
                    J_m=4.6063e-7,  # motor moment of inertia [kg*m**2]
                    k_m=0.0077,  # motor torque constant [N*m/A] = back-EMF constant [V*s/rad]
                    R_m=2.6,  # motor armature resistance
                    eta_m=0.69,  # motor efficiency [-]
                    B_eq=0.015,  # equivalent viscous damping coefficient w.r.t. load [N*m*s/rad]
                    c_frict=0.05,  # viscous friction coefficient [N*s/m]
                    V_thold_x_pos=V_tholds['V_thold_x_pos'],  # voltage required to move the x servo in positive dir
                    V_thold_x_neg=V_tholds['V_thold_x_neg'],  # voltage required to move the x servo in negative dir
                    V_thold_y_pos=V_tholds['V_thold_y_pos'],  # voltage required to move the y servo in positive dir
                    V_thold_y_neg=V_tholds['V_thold_y_neg'],  # voltage required to move the y servo in negative dir
                    offset_th_x=0.,  # angular offset of the x axis motor shaft [rad]
                    offset_th_y=0.)  # angular offset of the y axis motor shaft [rad]

    def _calc_constants(self):
        l_plate = self.domain_param['l_plate']
        m_ball = self.domain_param['m_ball']
        r_ball = self.domain_param['r_ball']
        eta_g = self.domain_param['eta_g']
        eta_m = self.domain_param['eta_m']
        K_g = self.domain_param['K_g']
        J_m = self.domain_param['J_m']
        J_l = self.domain_param['J_l']
        r_arm = self.domain_param['r_arm']
        k_m = self.domain_param['k_m']
        R_m = self.domain_param['R_m']
        B_eq = self.domain_param['B_eq']

        self.J_ball = 2./5*m_ball*r_ball**2  # inertia of the ball [kg*m**2]
        self.J_eq = eta_g*K_g**2*J_m + J_l  # equivalent moment of inertia [kg*m**2]
        self.c_kin = 2.*r_arm/l_plate  # coefficient for the rod-plate kinematic
        self.A_m = eta_g*K_g*eta_m*k_m/R_m
        self.B_eq_v = eta_g*K_g**2*eta_m*k_m**2/R_m + B_eq
        self.zeta = m_ball*r_ball**2 + self.J_ball  # combined moment of inertial for the ball

    def _state_from_init(self, init_state):
        state = np.zeros(8)
        state[2:4] = init_state[:2]
        state[6:8] = init_state[2:]
        return state

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        obs = super().reset(init_state=init_state, domain_param=domain_param)

        # Reset the plate angles
        if self._simplified_dyn:
            self.plate_angs = np.zeros(2)  # actually not necessary since not used
        else:
            offset_th_x = self.domain_param['offset_th_x']
            offset_th_y = self.domain_param['offset_th_y']
            # Get the plate angles from inverse kinematics for initial pose
            self.plate_angs[0] = self._kin(self.state[0] + offset_th_x)
            self.plate_angs[1] = self._kin(self.state[1] + offset_th_y)

        # Return perfect observation
        return obs

    def _step_dynamics(self, act: np.ndarray):
        g = self.domain_param['g']
        m_ball = self.domain_param['m_ball']
        r_ball = self.domain_param['r_ball']
        c_frict = self.domain_param['c_frict']
        V_thold_x_neg = self.domain_param['V_thold_x_neg']
        V_thold_x_pos = self.domain_param['V_thold_x_pos']
        V_thold_y_neg = self.domain_param['V_thold_y_neg']
        V_thold_y_pos = self.domain_param['V_thold_y_pos']
        offset_th_x = self.domain_param['offset_th_x']
        offset_th_y = self.domain_param['offset_th_y']

        if not self._simplified_dyn:
            # Apply a voltage dead zone (i.e. below a certain amplitude the system does not move). This is a very
            # simple model of static friction. Experimentally evaluated the voltage required to get the plate moving.
            if V_thold_x_neg <= act[0] <= V_thold_x_pos:
                act[0] = 0
            if V_thold_y_neg <= act[1] <= V_thold_y_pos:
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

        th_x_ddot = (self.A_m*act[0] - self.B_eq_v*th_x_dot)/self.J_eq
        th_y_ddot = (self.A_m*act[1] - self.B_eq_v*th_y_dot)/self.J_eq

        '''
        THIS IS TIME INTENSIVE
        if not self._simplified_dyn:
            # Get the plate angles from inverse kinematics
            self.plate_angs[0] = self._kin(self.state[0] + self.offset_th_x)
            self.plate_angs[1] = self._kin(self.state[1] + self.offset_th_y)
        '''

        # Plate (not part of the state since it is a redundant information)
        # The definition of th_y is opposing beta, i.e.
        a = self.plate_angs[0]  # plate's angle around the y axis (alpha)
        b = self.plate_angs[1]  # plate's angle around the x axis (beta)
        a_dot = self.c_kin*th_x_dot*np.cos(th_x)/np.cos(a)  # plate's angular velocity around the y axis (alpha)
        b_dot = self.c_kin*-th_y_dot*np.cos(-th_y)/np.cos(b)  # plate's angular velocity around the x axis (beta)
        # Plate's angular accelerations (unused for simplified_dyn = True)
        a_ddot = 1./np.cos(a)*(self.c_kin*(th_x_ddot*np.cos(th_x) - th_x_dot**2*np.sin(th_x)) + a_dot**2*np.sin(a))
        b_ddot = 1./np.cos(b)*(self.c_kin*(-th_y_ddot*np.cos(th_y) - (-th_y_dot)**2*np.sin(-th_y)) + b_dot**2*np.sin(b))

        # kinematics: sin(a) = self.c_kin * sin(th_x)
        if self._simplified_dyn:
            # Ball dynamic without friction and Coriolis forces
            x_ddot = self.c_kin*m_ball*g*r_ball**2*np.sin(th_x)/self.zeta  # symm inertia
            y_ddot = self.c_kin*m_ball*g*r_ball**2*np.sin(th_y)/self.zeta  # symm inertia
        else:
            # Ball dynamic with friction and Coriolis forces
            x_ddot = (- c_frict*x_dot*r_ball**2  # friction
                      - self.J_ball*r_ball*a_ddot  # plate influence
                      + m_ball*x*a_dot**2*r_ball**2  # centripetal
                      + self.c_kin*m_ball*g*r_ball**2*np.sin(th_x)  # gravity
                      )/self.zeta
            y_ddot = (- c_frict*y_dot*r_ball**2  # friction
                      - self.J_ball*r_ball*b_ddot  # plate influence
                      + m_ball*y*(-b_dot)**2*r_ball**2  # centripetal
                      + self.c_kin*m_ball*g*r_ball**2*np.sin(th_y)  # gravity
                      )/self.zeta

        # Integration step (symplectic Euler)
        self.state[4:] += np.array([th_x_ddot, th_y_ddot, x_ddot, y_ddot])*self._dt  # next velocity
        self.state[:4] += self.state[4:]*self._dt  # next position

        # Integration step (forward Euler)
        self.plate_angs += np.array([a_dot, b_dot])*self._dt  # just for debugging when simplified dynamics

    def _init_anim(self):
        import vpython as vp

        l_plate = self.domain_param['l_plate']
        m_ball = self.domain_param['m_ball']
        r_ball = self.domain_param['r_ball']
        d_plate = 0.01  # only for animation

        # Init render objects on first call
        self._anim['canvas'] = vp.canvas(width=800, height=800, title="Quanser Ball-Balancer")
        self._anim['ball'] = vp.sphere(
            pos=vp.vec(self.state[2], self.state[3], r_ball + d_plate/2.),
            radius=r_ball,
            mass=m_ball,
            color=vp.color.red,
            canvas=self._anim['canvas']
        )
        self._anim['plate'] = vp.box(
            pos=vp.vec(0, 0, 0),
            size=vp.vec(l_plate, l_plate, d_plate),
            color=vp.color.green,
            canvas=self._anim['canvas']
        )
        self._anim['null_plate'] = vp.box(
            pos=vp.vec(0, 0, 0),
            size=vp.vec(l_plate*1.1, l_plate*1.1, d_plate/10),
            color=vp.color.cyan,
            opacity=0.5,  # 0 is fully transparent
            canvas=self._anim['canvas']
        )

    def _update_anim(self):
        import vpython as vp

        g = self.domain_param['g']
        l_plate = self.domain_param['l_plate']
        m_ball = self.domain_param['m_ball']
        r_ball = self.domain_param['r_ball']
        eta_g = self.domain_param['eta_g']
        eta_m = self.domain_param['eta_m']
        K_g = self.domain_param['K_g']
        J_m = self.domain_param['J_m']
        J_l = self.domain_param['J_l']
        r_arm = self.domain_param['r_arm']
        k_m = self.domain_param['k_m']
        R_m = self.domain_param['R_m']
        B_eq = self.domain_param['B_eq']
        c_frict = self.domain_param['c_frict']
        V_thold_x_neg = self.domain_param['V_thold_x_neg']
        V_thold_x_pos = self.domain_param['V_thold_x_pos']
        V_thold_y_neg = self.domain_param['V_thold_y_neg']
        V_thold_y_pos = self.domain_param['V_thold_y_pos']
        offset_th_x = self.domain_param['offset_th_x']
        offset_th_y = self.domain_param['offset_th_y']
        d_plate = 0.01  # only for animation
        #  Compute plate orientation
        a_vp = -self.plate_angs[0]  # plate's angle around the y axis (alpha)
        b_vp = self.plate_angs[1]  # plate's angle around the x axis (beta)

        # Axis runs along the x direction
        self._anim['plate'].size = vp.vec(l_plate, l_plate, d_plate)
        self._anim['plate'].axis = vp.vec(vp.cos(a_vp), 0, vp.sin(a_vp))*float(l_plate)
        # Up runs along the y direction (vpython coordinate system)
        self._anim['plate'].up = vp.vec(0, vp.cos(b_vp), vp.sin(b_vp))

        # Get ball position
        x = self.state[2]  # along the x axis
        y = self.state[3]  # along the y axis

        self._anim['ball'].pos = vp.vec(
            x*vp.cos(a_vp),
            y*vp.cos(b_vp),
            r_ball + x*vp.sin(a_vp) + y*vp.sin(b_vp) + vp.cos(a_vp)*d_plate/2.
        )
        self._anim['ball'].radius = r_ball

        # Set caption text
        self._anim['canvas'].caption = f"""
            x-axis is pos to the right, y-axis is pos up
            Commanded voltage: x servo : {self._curr_act[0] : 1.2f}, y servo : {self._curr_act[1] : 1.2f}
            Plate angle around x axis: {self.plate_angs[1]*180/np.pi : 2.2f}
            Plate angle around y axis: {self.plate_angs[0]*180/np.pi : 2.2f}
            Shaft angles: {self.state[0]*180/np.pi : 2.2f}, {self.state[1]*180/np.pi : 2.2f}
            Ball position: {x : 1.3f}, {y : 1.3f}
            g: {g : 1.3f}
            m_ball: {m_ball : 1.3f}
            r_ball: {r_ball : 1.3f}
            r_arm: {r_arm : 1.3f}
            l_plate: {l_plate : 1.3f}
            K_g: {K_g : 2.2f}
            J_m: {J_m : 1.7f}
            J_l: {J_l : 1.6f}
            eta_g: {eta_g : 1.3f}
            eta_m: {eta_m : 1.3f}
            k_mt: {k_m : 1.3f}
            R_m: {R_m : 1.3f}
            B_eq: {B_eq : 1.3f}
            c_frict: {c_frict : 1.3f}
            V_thold_x_pos: {V_thold_x_pos : 2.3f}
            V_thold_x_neg: {V_thold_x_neg : 2.3f}
            V_thold_y_pos: {V_thold_y_pos : 2.3f}
            V_thold_y_neg: {V_thold_y_neg : 2.3f}
            offset_th_x: {offset_th_x : 2.3f}
            offset_th_y: {offset_th_y : 2.3f}
            """


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
        Serializable._init(self, locals())

        self._qbb = qbb
        self.num_opt_iter = num_opt_iter
        self.render_mode = render_mode

        self.r = float(self._qbb.domain_param['r_arm'])
        self.l = float(self._qbb.domain_param['l_plate']/2.)
        self.d = 0.10  # [m] roughly measured

        # Visualization
        if render_mode.video:
            self.fig, self.ax = plt.subplots(figsize=(5, 5))
            self.ax.set_xlim(-0.5*self.r, 1.2*(self.r + self.l))
            self.ax.set_ylim(-1.0*self.d, 2*self.d)
            self.ax.set_aspect('equal')
            self.line1, = self.ax.plot([0, 0], [0, 0], marker='o')
            self.line2, = self.ax.plot([0, 0], [0, 0], marker='o')
            self.line3, = self.ax.plot([0, 0], [0, 0], marker='o')

    def __call__(self, th):
        """
        Compute the inverse kinematics of the Quanser 2 DoF Ball-Balancer for one DoF

        :param th: angle of the servo (x or y axis)
        :return: plate angle al pha or beta
        """
        if not isinstance(th, to.Tensor):
            th = to.tensor(th, dtype=to.get_default_dtype(), requires_grad=False)

        # Update the lengths, e.g. if the domain has been randomized
        # Need to use float() since the parameters might be 0d-arrays
        self.r = float(self._qbb.domain_param['r_arm'])
        self.l = float(self._qbb.domain_param['l_plate']/2.)
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
        rod_len = to.sqrt((tip[0] - self.r*to.cos(th))**2 + (tip[1] - self.r*to.sin(th))**2)
        half_palte = to.sqrt((tip[0] - self.r - self.l)**2 + (tip[1] - self.d)**2)

        return (rod_len - self.d)**2 + (half_palte - self.l)**2

    def plate_ang(self, tip):
        """
        Compute plate angle (alpha or beta) from the rod tip position which has been calculated from servo shaft angle
        (th_x or th_y) before.
        :return tip: 2D position of the rod tip in the sagittal plane (from the optimizer)
        """
        ang = np.pi/2. - to.atan2(self.r + self.l - tip[0], tip[1] - self.d)
        return float(ang)

    def render(self, th, tip):
        """
        Visualize using pyplot

        :param th: angle of the servo
        :param tip: 2D position of the rod tip in the sagittal plane (from the optimizer)
        """
        A = [0, 0]
        B = [self.r*np.cos(th), self.r*np.sin(th)]
        C = [tip[0], tip[1]]
        D = [self.r + self.l, self.d]

        self.line1.set_data([A[0], B[0]], [A[1], B[1]])
        self.line2.set_data([B[0], C[0]], [B[1], C[1]])
        self.line3.set_data([C[0], D[0]], [C[1], D[1]])

    def _get_state(self, state_dict):
        state_dict['r'] = self.r
        state_dict['l'] = self.l

    def _set_state(self, state_dict, copying=False):
        self.r = state_dict['r']
        self.l = state_dict['l']
