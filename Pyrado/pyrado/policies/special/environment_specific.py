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

import math
import time
from typing import Union

import numpy as np
import torch as to
import torch.nn as nn

import pyrado
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.policies.features import FeatureStack, identity_feat
from pyrado.policies.feed_back.linear import LinearPolicy
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

    def reset(self, state_des: Union[np.ndarray, to.Tensor] = None):
        """
        Set the controller's desired state.

        :param state_des: tensor of desired x and y ball position [m], or None to keep the current desired state
        """
        if state_des is not None:
            if isinstance(state_des, to.Tensor):
                pass
            elif isinstance(state_des, np.ndarray):
                self.state_des = to.from_numpy(state_des).type(to.get_default_dtype())
            else:
                raise pyrado.TypeErr(given=state_des, expected_type=[to.Tensor, np.ndarray])
            self.state_des = state_des.clone()


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

        J_pole = self.dp_nom["l_pole"] ** 2 * self.dp_nom["m_pole"] / 3.0
        J_eq = (
            self.dp_nom["m_cart"]
            + (self.dp_nom["eta_g"] * self.dp_nom["K_g"] ** 2 * self.dp_nom["J_m"]) / self.dp_nom["r_mp"] ** 2
        )

        # Energy terms: E_pot(0) = 0; E_pot(pi) = E_pot(-pi) = 2 mgl
        E_kin = J_pole / 2.0 * theta_dot ** 2
        E_pot = self.dp_nom["m_pole"] * self.dp_nom["g"] * self.dp_nom["l_pole"] * (1 - cos_th)
        E_ref = 2.0 * self.dp_nom["m_pole"] * self.dp_nom["g"] * self.dp_nom["l_pole"]

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

        act = (J_eq * self.dp_nom["R_m"] * self.dp_nom["r_mp"] * u) / (
            self.dp_nom["eta_g"] * self.dp_nom["K_g"] * self.dp_nom["eta_m"] * self.dp_nom["k_m"]
        ) + self.dp_nom["K_g"] * self.dp_nom["k_m"] * x_dot / self.dp_nom["r_mp"]

        # Return the clipped action
        return act.view(1)  # such that when act is later converted to numpy it does not become a float


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
        self.e_ctrl = QQubeEnergyCtrl(env_spec, ref_energy, energy_gain, energy_th_gain, acc_max)
        self.pd_ctrl = QQubePDCtrl(env_spec, pd_gains, al_des=math.pi)

    def pd_enabled(self, cos_al: [float, to.Tensor]) -> bool:
        """
        Check if the PD-controller should be enabled based oin a predefined threshold on the alpha angle.

        :param cos_al: cosine of the pendulum pole angle
        :return: bool if condition is met
        """
        cos_al_delta = 1.0 + to.cos(to.tensor(math.pi - self.alpha_max_pd_enable))
        return bool(to.abs(1.0 + cos_al) < cos_al_delta)

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

        if self.pd_enabled(cos_al):
            s[1] = s[1] % (2 * math.pi)  # alpha can have multiple revolutions
            return self.pd_ctrl(s)
        else:
            return self.e_ctrl(s)


class QQubeEnergyCtrl(Policy):
    """Energy-based controller used to swing the pendulum up"""

    def __init__(
        self,
        env_spec: EnvSpec,
        ref_energy: float,
        energy_gain: float,
        th_gain: float,
        acc_max: float,
        use_cuda: bool = False,
    ):
        """
        Constructor

        :param env_spec: environment specification
        :param ref_energy: reference energy level [J]
        :param energy_gain: P-gain on the energy [m/s/J]
        :param th_gain: P-gain on angle theta
        :param acc_max: maximum linear acceleration of the pendulum pivot [m/s**2]
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
        self.dp_nom = QQubeSwingUpSim.get_nominal_domain_param()

        # Default initialization
        self.init_param(None)

    @property
    def E_ref(self):
        return to.exp(self._log_E_ref)

    # @E_ref.setter
    # def E_ref(self, new_E_ref: float):
    #     self._log_E_ref = to.log(new_E_ref)

    @property
    def E_gain(self):
        r"""Called $\mu$ by Quanser."""
        return to.exp(self._log_E_gain)

    # @E_gain.setter
    # def E_gain(self, new_mu):
    #     r""" Called $\mu$ by Quanser."""
    #     self._log_E_gain = to.log(new_mu)

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
        J_pole = self.dp_nom["Mp"] * self.dp_nom["Lp"] ** 2 / 12.0
        E_kin = 0.5 * J_pole * ald ** 2
        E_pot = 0.5 * self.dp_nom["Mp"] * self.dp_nom["g"] * self.dp_nom["Lp"] * (1.0 - to.cos(al))
        E = E_kin + E_pot

        # Compute clipped action
        u = self.E_gain * (E - self.E_ref) * to.sign(ald * to.cos(al)) - self._th_gain * th
        acc = clamp_symm(u, self.acc_max)
        trq = self.dp_nom["Mr"] * self.dp_nom["Lr"] * acc
        volt = self.dp_nom["Rm"] / self.dp_nom["km"] * trq
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


def get_lin_ctrl(env: SimEnv, ctrl_type: str, ball_z_dim_mismatch: bool = True) -> LinearPolicy:
    """
    Construct a linear controller specified by its controller gains.
    Parameters for BallOnPlate5DSim by Markus Lamprecht (clipped gains < 1e-5 to 0).

    :param env: environment
    :param ctrl_type: type of the controller: 'lqr', or 'h2'
    :param ball_z_dim_mismatch: only useful for BallOnPlate5DSim
                                set to True if the given controller dos not have the z component (relative position)
                                of the ball in the state vector, i.e. state is 14-dim instead of 16-dim
    :return: controller compatible with Pyrado Policy
    """
    from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate5DSim

    if isinstance(inner_env(env), BallOnPlate5DSim):
        # Get the controller gains (K-matrix)
        if ctrl_type.lower() == "lqr":
            ctrl_gains = to.tensor(
                [
                    [0.1401, 0, 0, 0, -0.09819, -0.1359, 0, 0.545, 0, 0, 0, -0.01417, -0.04427, 0],
                    [0, 0.1381, 0, 0.2518, 0, 0, -0.2142, 0, 0.5371, 0, 0.03336, 0, 0, -0.1262],
                    [0, 0, 0.1414, 0.0002534, 0, 0, -0.0002152, 0, 0, 0.5318, 0, 0, 0, -0.0001269],
                    [0, -0.479, -0.0004812, 39.24, 0, 0, -15.44, 0, -1.988, -0.001934, 9.466, 0, 0, -13.14],
                    [0.3039, 0, 0, 0, 25.13, 15.66, 0, 1.284, 0, 0, 0, 7.609, 6.296, 0],
                ]
            )

        elif ctrl_type.lower() == "h2":
            ctrl_gains = to.tensor(
                [
                    [
                        -73.88,
                        -2.318,
                        39.49,
                        -4.270,
                        12.25,
                        0.9779,
                        0.2564,
                        35.11,
                        5.756,
                        0.8661,
                        -0.9898,
                        1.421,
                        3.132,
                        -0.01899,
                    ],
                    [
                        -24.45,
                        0.7202,
                        -10.58,
                        2.445,
                        -0.6957,
                        2.1619,
                        -0.3966,
                        -61.66,
                        -3.254,
                        5.356,
                        0.1908,
                        12.88,
                        6.142,
                        -0.3812,
                    ],
                    [
                        -101.8,
                        -9.011,
                        64.345,
                        -5.091,
                        17.83,
                        -2.636,
                        0.9506,
                        -44.28,
                        3.206,
                        37.59,
                        2.965,
                        -32.65,
                        -21.68,
                        -0.1133,
                    ],
                    [
                        -59.56,
                        1.56,
                        -0.5794,
                        26.54,
                        -2.503,
                        3.827,
                        -7.534,
                        9.999,
                        1.143,
                        -16.96,
                        8.450,
                        -5.302,
                        4.620,
                        -10.32,
                    ],
                    [
                        -107.1,
                        0.4359,
                        19.03,
                        -9.601,
                        20.33,
                        10.36,
                        0.2285,
                        -74.98,
                        -2.136,
                        7.084,
                        -1.240,
                        62.62,
                        33.66,
                        1.790,
                    ],
                ]
            )

        else:
            raise pyrado.ValueErr(given=ctrl_type, eq_constraint="'lqr' or 'h2'")

        # Compensate for the mismatching different state definition
        if ball_z_dim_mismatch:
            ctrl_gains = insert_tensor_col(ctrl_gains, 7, to.zeros((5, 1)))  # ball z position
            ctrl_gains = insert_tensor_col(ctrl_gains, -1, to.zeros((5, 1)))  # ball z velocity

    elif isinstance(inner_env(env), QBallBalancerSim):
        # Get the controller gains (K-matrix)
        if ctrl_type.lower() == "pd":
            # Quanser gains (the original Quanser controller includes action clipping)
            ctrl_gains = -to.tensor(
                [[-14.0, 0, -14 * 3.45, 0, 0, 0, -14 * 2.11, 0], [0, -14.0, 0, -14 * 3.45, 0, 0, 0, -14 * 2.11]]
            )

        elif ctrl_type.lower() == "lqr":
            # Since the control module can by tricky to install (recommended using anaconda), we only load it if needed
            import control  # pylint: disable=import-error,wrong-import-order

            # System modeling
            A = np.zeros((env.obs_space.flat_dim, env.obs_space.flat_dim))
            A[: env.obs_space.flat_dim // 2, env.obs_space.flat_dim // 2 :] = np.eye(env.obs_space.flat_dim // 2)
            A[4, 4] = -env.B_eq_v / env.J_eq
            A[5, 5] = -env.B_eq_v / env.J_eq
            A[6, 0] = env.c_kin * env.m_ball * env.g * env.r_ball ** 2 / env.zeta
            A[6, 6] = -env.c_kin * env.r_ball ** 2 / env.zeta
            A[7, 1] = env.c_kin * env.m_ball * env.g * env.r_ball ** 2 / env.zeta
            A[7, 7] = -env.c_kin * env.r_ball ** 2 / env.zeta
            B = np.zeros((env.obs_space.flat_dim, env.act_space.flat_dim))
            B[4, 0] = env.A_m / env.J_eq
            B[5, 1] = env.A_m / env.J_eq
            # C = np.zeros((env.obs_space.flat_dim // 2, env.obs_space.flat_dim))
            # C[:env.obs_space.flat_dim // 2, :env.obs_space.flat_dim // 2] = np.eye(env.obs_space.flat_dim // 2)
            # D = np.zeros((env.obs_space.flat_dim // 2, env.act_space.flat_dim))

            # Get the weighting matrices from the environment
            Q = env.task.rew_fcn.Q
            R = env.task.rew_fcn.R
            # Q = np.diag([1e2, 1e2, 5e2, 5e2, 1e-2, 1e-2, 1e+1, 1e+1])

            # Solve the continuous time Riccati eq
            K, _, _ = control.lqr(A, B, Q, R)  # for discrete system pass dt
            ctrl_gains = to.from_numpy(K).to(to.get_default_dtype())
        else:
            raise pyrado.ValueErr(given=ctrl_type, eq_constraint="'pd', 'lqr'")

    else:
        raise pyrado.TypeErr(given=inner_env(env), expected_type=BallOnPlate5DSim)

    # Reconstruct the controller
    feats = FeatureStack([identity_feat])
    ctrl = LinearPolicy(env.spec, feats)
    ctrl.init_param(-1 * ctrl_gains)  # in classical control it is u = -K*x; here a = psi(s)*s
    return ctrl


def wam_jsp_7dof_sin(t: float, flip_sign: bool = False):
    """
    A sin-based excitation function for the 7-DoF WAM, describing desired a desired joint angle offset and its velocity
    at every point in time

    :param t: time
    :param flip_sign: if `True`, flip the sign
    :return: joint angle positions and velocities
    """
    flip_sign = int(flip_sign)
    return -(1 ** flip_sign) * np.array(
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
