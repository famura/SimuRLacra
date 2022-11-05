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
import functools
import math
import time

import numpy as np
import torch as to
import torch.nn as nn

import pyrado
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.policies.features import FeatureStack, identity_feat
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.policies.feed_forward.playback import PlaybackPolicy
from pyrado.policies.feed_forward.time import TimePolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.math import clamp_symm
from pyrado.utils.tensor import insert_tensor_col


class QBallBalancerPDCtrl(Policy):
    """
    PD-controller for the Quanser Ball Balancer.
    The only but significant difference of this controller to the other PD controller is the clipping of the actions.

    .. note::
        This class's desired state specification deviates from the Pyrado policies which interact with a `Task`.
    """

    name: str = "qbb-pd"

    def __init__(
        self,
        env_spec: EnvSpec,
        state_des: to.Tensor = to.zeros(2),
        kp: to.Tensor = None,
        kd: to.Tensor = None,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param env_spec: environment specification
        :param state_des: tensor of desired x and y ball position [m]
        :param kp: 2x2 tensor of constant controller feedback coefficients for error [V/m]
        :param kd: 2x2 tensor of constant controller feedback coefficients for error time derivative [Vs/m]
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(env_spec, use_cuda)

        self.state_des = state_des
        self.limit_rad = 0.52360  # limit for angle command; see the saturation block in the Simulink model
        self.kp_servo = 14.0  # P-gain for the servo angle; see the saturation block the Simulink model
        self.Kp, self.Kd = None, None

        # Default initialization
        self.init_param(kp, kd)

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Calculate the controller output.

        :param obs: observation from the environment
        :return act: controller output [V]
        """
        th_x, th_y, x, y, _, _, x_dot, y_dot = obs

        err = to.tensor([self.state_des[0] - x, self.state_des[1] - y])
        err_dot = to.tensor([0.0 - x_dot, 0.0 - y_dot])
        th_des = self.Kp.mv(err) + self.Kd.mv(err_dot)

        # Saturation for desired angular position
        th_des = to.clamp(th_des, -self.limit_rad, self.limit_rad)
        err_th = th_des - to.tensor([th_x, th_y])

        # Return action, see "Actuator Electrical Dynamics" block in [1]
        return err_th * self.kp_servo

    def init_param(self, kp: to.Tensor = None, kd: to.Tensor = None, verbose: bool = False, **kwargs):
        """
        Initialize controller parameters.

        :param kp: 2x2 tensor of constant controller feedback coefficients for error [V/m]
        :param kd: 2x2 tensor of constant controller feedback coefficients for error time derivative [Vs/m]
        :param verbose: print the controller's gains
        """
        self.Kp = to.diag(to.tensor([3.45, 3.45])) if kp is None else kp
        self.Kd = to.diag(to.tensor([2.11, 2.11])) if kd is None else kd
        if not self.Kp.shape == (2, 2):
            raise pyrado.ShapeErr(given=self.Kp, expected_match=(2, 2))
        if not self.Kd.shape == (2, 2):
            raise pyrado.ShapeErr(given=self.Kd, expected_match=(2, 2))

        if verbose:
            print(f"Set Kp to\n{self.Kp.numpy()}\nand Kd to\n{self.Kd.numpy()}")

    def reset(self, **kwargs):
        """Set the domain parameters defining the controller's model using a dict called `domain_param`."""
        state_des = kwargs.get("state_des", None)  # do nothing if state_des not given
        if state_des is not None:
            if isinstance(state_des, to.Tensor):
                self.state_des = state_des.clone()
            elif isinstance(state_des, np.ndarray):
                self.state_des = to.from_numpy(state_des).type(to.get_default_dtype())
            else:
                raise pyrado.TypeErr(given=state_des, expected_type=(to.Tensor, np.ndarray))


class QCartPoleSwingUpAndBalanceCtrl(Policy):
    """Swing-up and balancing controller for the Quanser Cart-Pole"""

    name: str = "qcp-sub"

    def __init__(self, env_spec: EnvSpec, long: bool = False, use_cuda: bool = False):
        """
        Constructor

        :param env_spec: environment specification
        :param long: flag for long or short pole
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(env_spec, use_cuda)

        self.long = long
        self.pd_control = False
        self.pd_activated = False
        self.dp_nom = QCartPoleSim.get_nominal_domain_param(self.long)

        # Initial parameters
        self._log_u_max_init = to.log(to.tensor(18.0))
        if long:
            self._log_K_pd_init = to.log(to.tensor([41.833, 189.8393, 47.8483, 28.0941]))
        else:
            self._log_K_pd_init = to.log(
                to.tensor([41.0, 200.0, 55.0, 16.0])
            )  # former: [+41.8, 173.4, +46.1, 16.2], [34.1, 118.0, 43.4, 18.1]
            self._log_k_e_init = to.log(to.tensor(17.0))  # former: 24.5, 36.5, 19.5
            self._log_k_p_init = to.log(to.tensor(8.0))  # former: 4.0, 8.5, 2.25

        # Define parameters
        self._log_u_max = nn.Parameter(to.empty_like(self._log_u_max_init), requires_grad=True)
        self._log_K_pd = nn.Parameter(to.empty_like(self._log_K_pd_init), requires_grad=True)
        if not long:
            self._log_k_e = nn.Parameter(to.empty_like(self._log_k_e_init), requires_grad=True)
            self._log_k_p = nn.Parameter(to.empty_like(self._log_k_p_init), requires_grad=True)

        # Default initialization
        self.init_param(None)

    @property
    def u_max(self):
        return to.exp(self._log_u_max)

    @property
    def k_e(self):
        return to.exp(self._log_k_e)

    @property
    def k_p(self):
        return to.exp(self._log_k_p)

    @property
    def K_pd(self):
        if self.long:
            return to.exp(self._log_K_pd) * to.tensor([-1, 1, -1, 1])
        else:
            return to.exp(self._log_K_pd) * to.tensor([1, -1, 1, -1])  # the gains related to theta need to be negative

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is not None:
            self.param_values = init_values

        else:
            self._log_u_max.data = self._log_u_max_init
            self._log_K_pd.data = self._log_K_pd_init
            if not self.long:
                self._log_k_e.data = self._log_k_e_init
                self._log_k_p.data = self._log_k_p_init

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Calculate the controller output.

        :param obs: observation from the environment
        :return act: controller output [V]
        """
        x, sin_th, cos_th, x_dot, theta_dot = obs
        theta = to.atan2(sin_th, cos_th)
        alpha = (theta - math.pi) if theta > 0 else (theta + math.pi)

        J_pole = self.dp_nom["pole_length"] ** 2 * self.dp_nom["pole_mass"] / 3.0
        J_eq = (
            self.dp_nom["cart_mass"]
            + (self.dp_nom["gear_efficiency"] * self.dp_nom["gear_ratio"] ** 2 * self.dp_nom["motor_inertia"])
            / self.dp_nom["pinion_radius"] ** 2
        )

        # Energy terms: E_pot(0) = 0; E_pot(pi) = E_pot(-pi) = 2 mgl
        E_kin = J_pole / 2.0 * theta_dot**2
        E_pot = self.dp_nom["pole_mass"] * self.dp_nom["gravity_const"] * self.dp_nom["pole_length"] * (1 - cos_th)
        E_ref = 2.0 * self.dp_nom["pole_mass"] * self.dp_nom["gravity_const"] * self.dp_nom["pole_length"]

        if to.abs(alpha) < 0.1745 or self.pd_control:
            # Stabilize at the top
            self.pd_activated = True
            u = self.K_pd.dot(to.tensor([x, alpha, x_dot, theta_dot]))
        else:
            # Swing up
            u = self.k_e * (E_kin + E_pot - E_ref) * to.sign(theta_dot * cos_th) + self.k_p * (0.0 - x)
            u = clamp_symm(u, self.u_max)

            if self.pd_activated:
                self.pd_activated = False

        act = (J_eq * self.dp_nom["motor_resistance"] * self.dp_nom["pinion_radius"] * u) / (
            self.dp_nom["gear_efficiency"]
            * self.dp_nom["gear_ratio"]
            * self.dp_nom["motor_efficiency"]
            * self.dp_nom["motor_back_emf"]
        ) + self.dp_nom["gear_ratio"] * self.dp_nom["motor_back_emf"] * x_dot / self.dp_nom["pinion_radius"]

        # Return the clipped action
        return act.view(1)  # such that when act is later converted to numpy it does not become a float


class QCartPoleGoToLimCtrl:
    """Controller for going to one of the joint limits (part of the calibration routine)"""

    def __init__(self, init_state: np.ndarray, positive: bool = True):
        """
        Constructor

        :param init_state: initial state of the system
        :param positive: direction switch
        """
        self.done = False
        self.success = False
        self.x_init = init_state[0]
        self.x_lim = 0.0
        self.xd_max = 1e-4
        self.delta_x_min = 0.1
        self.sign = 1 if positive else -1
        self.u_max = self.sign * np.array([1.5])
        self._t0 = None
        self._t_max = 10.0
        self._t_min = 2.0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Go to joint limits by applying u_max and save limit value in th_lim.

        :param obs: observation from the environment
        :return: action
        """
        x, _, _, xd, _ = obs

        # Initialize time
        if self._t0 is None:
            self._t0 = time.time()

        # Compute voltage
        if (time.time() - self._t0) < self._t_min:
            # Go full speed before t_min
            u = self.u_max
        elif (time.time() - self._t0) > self._t_max:
            # Do nothing if t_max is elapsed
            u = np.zeros(1)
            self.success, self.done = False, True
        elif np.abs(xd) < self.xd_max:  # and np.abs(x - self.x_init) > self.delta_x_min:
            # Do nothing
            u = np.zeros(1)
            self.success, self.done = True, True
        else:
            u = self.u_max

        return u


class QQubeSwingUpAndBalanceCtrl(Policy):
    """Hybrid controller (QQubeEnergyCtrl, QQubePDCtrl) switching based on the pendulum pole angle alpha

    .. note::
        Extracted Quanser's values from q_qube2_swingup.mdl
    """

    name: str = "qq-sub"

    def __init__(
        self,
        env_spec: EnvSpec,
        ref_energy: float = 0.025,  # Quanser's value: 0.02
        energy_gain: float = 50.0,  # Quanser's value: 50
        energy_th_gain: float = 0.4,  # former: 0.4
        acc_max: float = 5.0,  # Quanser's value: 6
        alpha_max_pd_enable: float = 20.0,  # Quanser's value: 20
        pd_gains: to.Tensor = to.tensor([-2, 35, -1.5, 3]),  # Quanser's value: [-2, 35, -1.5, 3]
        reset_domain_param: bool = True,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param env_spec: environment specification
        :param ref_energy: reference energy level
        :param energy_gain: P-gain on the difference to the reference energy
        :param energy_th_gain: P-gain on angle theta for the Energy controller. This term does not exist in Quanser's
                               implementation. Its purpose it to keep the Qube from moving too much around the
                               vertical axis, i.e. prevent bouncing against the mechanical boundaries.
        :param acc_max: maximum acceleration
        :param alpha_max_pd_enable: angle threshold for enabling the PD -controller [deg]
        :param pd_gains: gains for the PD-controller
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU

        .. note::
            The controller's parameters strongly depend on the frequency at which it is operating.
        """
        super().__init__(env_spec, use_cuda)

        self.alpha_max_pd_enable = alpha_max_pd_enable / 180.0 * math.pi

        # Set up the energy and PD controller
        self.e_ctrl = QQubeEnergyCtrl(env_spec, ref_energy, energy_gain, energy_th_gain, acc_max, reset_domain_param)
        self.pd_ctrl = QQubePDCtrl(env_spec, pd_gains, al_des=math.pi)

    def reset(self, **kwargs):
        # Forward to the two controllers
        self.e_ctrl.reset(**kwargs)
        self.pd_ctrl.reset(**kwargs)

    def pd_enabled(self, cos_al: [float, to.Tensor]) -> bool:
        """
        Check if the PD-controller should be enabled based oin a predefined threshold on the alpha angle.

        :param cos_al: cosine of the pendulum pole angle
        :return: bool if condition is met
        """
        cos_al_delta = 1.0 + to.cos(to.tensor(math.pi - self.alpha_max_pd_enable))
        return to.abs(1.0 + cos_al) < cos_al_delta

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is not None:
            self.param_values = init_values

        else:
            # Forward to the individual controllers
            self.e_ctrl.init_param()
            self.pd_ctrl.init_param()

    def forward(self, obs: to.tensor):
        # Reconstruct the sate for the error-based controller
        sin_th, cos_th, sin_al, cos_al, th_d, al_d = obs
        s = to.stack([to.atan2(sin_th, cos_th), to.atan2(sin_al, cos_al), th_d, al_d])
        s[1] = s[1] % (2 * math.pi)  # alpha can have multiple revolutions

        if self.pd_enabled(cos_al):
            return self.pd_ctrl(s)
        else:
            return self.e_ctrl(s)


class QQubeEnergyCtrl(Policy):
    """Energy-based controller used to swing the Furuta pendulum up"""

    def __init__(
        self,
        env_spec: EnvSpec,
        ref_energy: float,
        energy_gain: float,
        th_gain: float,
        acc_max: float,
        reset_domain_param: bool = True,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param env_spec: environment specification
        :param ref_energy: reference energy level [J]
        :param energy_gain: P-gain on the energy [m/s/J]
        :param th_gain: P-gain on angle theta
        :param acc_max: maximum linear acceleration of the pendulum pivot [m/s**2]
        :param reset_domain_param: if `True` the domain parameters are reset if the they are present as a entry in the
                                   kwargs passed to `reset()`. If `False` they are ignored.
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(env_spec, use_cuda)

        # Initial parameters
        self._log_E_ref_init = to.log(to.tensor(ref_energy))
        self._log_E_gain_init = to.log(to.tensor(energy_gain))
        self._th_gain_init = to.tensor(th_gain)

        # Define parameters
        self._log_E_ref = nn.Parameter(to.empty_like(self._log_E_ref_init), requires_grad=True)
        self._log_E_gain = nn.Parameter(to.empty_like(self._log_E_gain_init), requires_grad=True)
        self._th_gain = nn.Parameter(to.empty_like(self._th_gain_init), requires_grad=True)

        self.acc_max = to.tensor(acc_max)
        self._domain_param = QQubeSwingUpSim.get_nominal_domain_param()
        self._reset_domain_param = reset_domain_param

        # Default initialization
        self.init_param(None)

    def reset(self, **kwargs):
        """If desired, set the domain parameters defining the controller's model using a dict called `domain_param`."""
        if self._reset_domain_param:
            domain_param = kwargs.get("domain_param", dict())  # do nothing if domain_param not given
            if isinstance(domain_param, dict):
                self._domain_param.update(domain_param)
            else:
                raise pyrado.TypeErr(given=domain_param, expected_type=dict)

    @property
    def E_ref(self):
        """Get the reference energy level."""
        return to.exp(self._log_E_ref)

    @property
    def E_gain(self):
        r"""Get the energy gain, called $\mu$ in the Quanser documentation."""
        return to.exp(self._log_E_gain)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is not None:
            self.param_values = init_values

        else:
            # Initialize with original parameters
            self._log_E_ref.data = self._log_E_ref_init
            self._log_E_gain.data = self._log_E_gain_init
            self._th_gain.data = self._th_gain_init

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Control step of energy-based controller which is used in the swing-up controller

        :param obs: observations pre-processed in the `forward` method of `QQubeSwingUpAndBalanceCtrl`
        :return: action
        """
        # Reconstruct partial state
        th, al, thd, ald = obs

        # Compute energies
        J_pole = self._domain_param["mass_pend_pole"] * self._domain_param["length_pend_pole"] ** 2 / 12.0
        E_kin = 0.5 * J_pole * ald**2
        E_pot = (
            0.5
            * self._domain_param["mass_pend_pole"]
            * self._domain_param["gravity_const"]
            * self._domain_param["length_pend_pole"]
            * (1.0 - to.cos(al))
        )
        E = E_kin + E_pot

        # Compute clipped action
        u = self.E_gain * (E - self.E_ref) * to.sign(ald * to.cos(al)) - self._th_gain * th
        acc = clamp_symm(u, self.acc_max)
        trq = self._domain_param["mass_rot_pole"] * self._domain_param["length_rot_pole"] * acc
        volt = self._domain_param["motor_resistance"] / self._domain_param["motor_back_emf"] * trq
        return volt.view(1)


class QQubePDCtrl(Policy):
    r"""
    PD-controller for the Quanser Qube.
    Drives Qube to $x_{des} = [\theta_{des}, \alpha_{des}, 0.0, 0.0]$.
    Flag done is set when $|x_des - x| < tol$.
    """

    def __init__(
        self,
        env_spec: EnvSpec,
        pd_gains: to.Tensor = to.tensor([4.0, 0, 1.0, 0]),
        th_des: float = 0.0,
        al_des: float = 0.0,
        tols: to.Tensor = to.tensor([1.5, 0.5, 0.1, 0.1], dtype=to.float64) / 180.0 * math.pi,
        use_cuda: bool = False,
    ):
        r"""
        Constructor

        :param env_spec: environment specification
        :param pd_gains: controller gains, the default values stabilize the pendulum at the center hanging down
        :param th_des: desired rotary pole angle [rad]
        :param al_des: desired pendulum pole angle [rad]
        :param tols: tolerances for the desired angles $\theta$ and $\alpha$ [rad]
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not isinstance(pd_gains, to.Tensor):
            raise pyrado.TypeErr(given=pd_gains, expected_type=to.Tensor)

        super().__init__(env_spec, use_cuda)

        # Initial parameters
        self._pd_gains_init = pd_gains

        # Define parameters
        self.pd_gains = nn.Parameter(to.empty_like(self._pd_gains_init), requires_grad=True)

        self.state_des = to.tensor([th_des, al_des, 0.0, 0.0])
        self.tols = to.as_tensor(tols)
        self.done = False

        # Default initialization
        self.init_param(None)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        if init_values is not None:
            self.param_values = init_values

        else:
            # Initialize with original parameters
            self.pd_gains.data = self._pd_gains_init

    def forward(self, meas: to.Tensor) -> to.Tensor:
        meas = meas.to(dtype=to.get_default_dtype())

        # Unpack the raw measurement (is not an observation)
        err = self.state_des - meas  # th, al, thd, ald

        if all(to.abs(err) <= self.tols):
            self.done = True
        elif all(to.abs(err) > self.tols) and self.state_des[0] == self.state_des[1] == 0.0:
            # In case of initializing the Qube, increase the P-gain over time. This is useful since the resistance from
            # the Qube's cable can be too strong for the PD controller to reach the steady state, like a fake I-gain.
            self.pd_gains.data = self.pd_gains + to.tensor([0.01, 0.0, 0.0, 0.0])  # no in-place op because of grad
            self.pd_gains.data = to.min(self.pd_gains, to.tensor([20.0, pyrado.inf, pyrado.inf, pyrado.inf]))

        # PD control
        return to.atleast_1d(self.pd_gains.dot(err))


class QQubeGoToLimCtrl:
    """Controller for going to one of the joint limits (part of the calibration routine)"""

    def __init__(self, positive: bool = True, cnt_done: int = 250):
        """
        Constructor

        :param positive: direction switch
        """
        self.done = False
        self.th_lim = pyrado.inf
        self.sign = 1 if positive else -1
        self.u_max = 0.9
        self.cnt = 0
        self.cnt_done = cnt_done

    def __call__(self, meas: to.Tensor) -> to.Tensor:
        """
        Go to joint limits by applying `u_max` and save limit value in `th_lim`.

        :param meas: sensor measurement
        :return: action
        """
        meas = meas.to(dtype=to.get_default_dtype())

        # Unpack the raw measurement (is not an observation)
        th = meas[0].item()

        if abs(th - self.th_lim) > 1e-6:
            # Recognized significant change in theta
            self.cnt = 0
            self.th_lim = th
        else:
            self.cnt += 1

        # Do this for cnt_done time steps
        self.done = self.cnt >= self.cnt_done
        return to.tensor([self.sign * self.u_max])


def create_pend_excitation_policy(env: PendulumSim, num_rollouts: int, f_sin: float = 1.0) -> PlaybackPolicy:
    """
    Create a policy that returns a previously recorded action time series.
    Used in the experiments of [1].

    .. seealso::
        [1] F. Muratore, T. Gruner, F. Wiese, B. Belousov, M. Gienger, J. Peters, "TITLE", VENUE, YEAR

    :param env: pendulum simulation environment
    :param num_rollouts: number of rollouts to store in the policy's buffer
    :param f_sin: frequency of the sinus [Hz]
    :return: policy with recorded action time series
    """

    def fcn_of_time(t: float):
        act = env.domain_param["torque_thold"] * np.sin(2 * np.pi * t * f_sin)
        return act.repeat(env.act_space.flat_dim)

    act_recordings = [
        [fcn_of_time(t) for t in np.arange(0, env.max_steps * env.dt, env.dt)] for _ in range(num_rollouts)
    ]
    return PlaybackPolicy(env.spec, act_recordings)


def _fcn_mg_joint_pos(t, q_init, q_end, t_strike_end):
    """Helper function for `create_mg_joint_pos_policy()` to fit the `TimePolicy` scheme"""
    return ((q_end - q_init) * min(t / t_strike_end, 1) + q_init) / 180 * math.pi


def create_mg_joint_pos_policy(env: SimEnv, t_strike_end: float = 0.5) -> TimePolicy:
    """
    Create a policy that executes the strike for mini golf by setting joint position commands.
    Used in the experiments of [1].

    .. seealso::
        [1] F. Muratore, T. Gruner, F. Wiese, B. Belousov, M. Gienger, J. Peters, "TITLE", VENUE, YEAR

    :param env: mini golf simulation environment
    :param t_strike_end:time when to finish the movement [s]
    :return: policy which executes the strike solely dependent on the time
    """
    q_init = to.tensor(
        [
            18.996253,
            -87.227101,
            74.149568,
            -75.577025,
            56.207369,
            -175.162794,
            -41.543793,
        ]
    )
    q_end = to.tensor(
        [
            8.628977,
            -93.443498,
            72.302435,
            -82.31844,
            52.146531,
            -183.896354,
            -51.560886,
        ]
    )

    return TimePolicy(
        env.spec, functools.partial(_fcn_mg_joint_pos, q_init=q_init, q_end=q_end, t_strike_end=t_strike_end), env.dt
    )


def wam_jsp_7dof_sin(t: float, flip_sign: bool = False):
    """
    A sin-based excitation function for the 7-DoF WAM, describing desired a desired joint angle offset and its velocity
    at every point in time

    :param t: time
    :param flip_sign: if `True`, flip the sign
    :return: joint angle positions and velocities
    """
    flip_sign = int(flip_sign)
    return -(1**flip_sign) * np.array(
        [
            0.5 * np.sin(2 * np.pi * t * 0.23),
            0.5 * np.sin(2 * np.pi * t * 0.51),
            0.5 * np.sin(2 * np.pi * t * 0.33),
            0.5 * np.sin(2 * np.pi * t * 0.41),
            0.5 * np.sin(2 * np.pi * t * 0.57),
            0.5 * np.sin(2 * np.pi * t * 0.63),
            0.5 * np.sin(2 * np.pi * t * 0.71),
            2 * np.pi * 0.23 * 0.5 * np.cos(2 * np.pi * t * 0.23),
            2 * np.pi * 0.51 * 0.5 * np.cos(2 * np.pi * t * 0.51),
            2 * np.pi * 0.33 * 0.5 * np.cos(2 * np.pi * t * 0.33),
            2 * np.pi * 0.23 * 0.5 * np.cos(2 * np.pi * t * 0.41),
            2 * np.pi * 0.57 * 0.5 * np.cos(2 * np.pi * t * 0.57),
            2 * np.pi * 0.63 * 0.5 * np.cos(2 * np.pi * t * 0.63),
            2 * np.pi * 0.71 * 0.5 * np.cos(2 * np.pi * t * 0.71),
        ]
    )
