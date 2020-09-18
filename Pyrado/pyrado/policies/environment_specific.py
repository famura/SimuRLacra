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
from typing import Union

import numpy as np
import time
import torch as to
import torch.nn as nn
from typing import Union

import pyrado
from pyrado.utils.data_types import EnvSpec
from pyrado.environments.sim_base import SimEnv
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environment_wrappers.utils import inner_env
from pyrado.policies.base import Policy
from pyrado.policies.features import FeatureStack, identity_feat, RBFFeat
from pyrado.policies.linear import LinearPolicy
from pyrado.utils.math import clamp_symm
from pyrado.utils.tensor import insert_tensor_col


class DualRBFLinearPolicy(LinearPolicy):
    """
    A linear policy with RBF features which are also used to get the derivative of the features. The use-case in mind
    is a simple policy which generates the joint position and joint velocity commands for the internal PD-controller
    of a robot (e.g. Barrett WAM). By re-using the RBF, we reduce the number of parameters, while we can at the same
    time get the velocity information from the features, i.e. the derivative of the normalized Gaussians.
    """

    name: str = 'dualrbf'

    def __init__(self,
                 spec: EnvSpec,
                 rbf_hparam: dict,
                 dim_mask: int = 2,
                 init_param_kwargs: dict = None,
                 use_cuda: bool = False):
        """
        Constructor

        :param spec: specification of environment
        :param rbf_hparam: hyper-parameters for the RBF-features, see `RBFFeat`
        :param dim_mask: number of RBF features to mask out at the beginning and the end of every dimension,
                         pass 1 to remove the first and the last features for the policy, pass 0 to use all
                         RBF features. Masking out RBFs makes sense if you want to obtain a smooth starting behavior.
        :param init_param_kwargs: additional keyword arguments for the policy parameter initialization
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        if not (0 <= dim_mask <= rbf_hparam['num_feat_per_dim']//2):
            raise pyrado.ValueErr(
                given=dim_mask, ge_constraint='0', le_constraint=f"{rbf_hparam['num_feat_per_dim']//2}"
            )

        # Construct the RBF features
        self._feats = RBFFeat(**rbf_hparam)

        # Call LinearPolicy's constructor (custom parts will be overridden later)
        super().__init__(spec, FeatureStack([self._feats]), init_param_kwargs, use_cuda)
        if not self._num_act%2 == 0:
            raise pyrado.ShapeErr(msg='DualRBFLinearPolicy only works with an even number of actions,'
                                      'since we are using the time derivative of the features to create the second'
                                      'half of the outputs. This is done to use forward() in order to obtain'
                                      'the joint position and the joint velocities.')

        # Override custom parts
        self._feats = RBFFeat(**rbf_hparam)
        self.dim_mask = dim_mask
        if self.dim_mask > 0:
            self.num_active_feat = self._feats.num_feat - 2*self.dim_mask*spec.obs_space.flat_dim
        else:
            self.num_active_feat = self._feats.num_feat
        self.net = nn.Linear(self.num_active_feat, self._num_act//2, bias=False)

        # Create mask to deactivate first and last feature of every input dimension
        self.feats_mask = to.ones(self._feats.centers.shape, dtype=to.bool)
        self.feats_mask[:self.dim_mask, :] = False
        self.feats_mask[-self.dim_mask:, :] = False
        self.feats_mask = self.feats_mask.t().reshape(-1, )  # reshape the same way as in RBFFeat

        # Call custom initialization function after PyTorch network parameter initialization
        init_param_kwargs = init_param_kwargs if init_param_kwargs is not None else dict()
        self.init_param(None, **init_param_kwargs)
        self.to(self.device)

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Evaluate the features at the given observation or use given feature values

        :param obs: observations from the environment
        :return: actions
        """
        obs = obs.to(self.device)
        batched = obs.ndimension() == 2  # number of dim is 1 if unbatched, dim > 2 is cought by features
        feats_val = self._feats(obs)
        feats_dot = self._feats.derivative(obs)

        if self.dim_mask > 0:
            # Mask out first and last feature of every input dimension
            feats_val = feats_val[:, self.feats_mask]
            feats_dot = feats_dot[:, self.feats_mask]

        # Inner product between policy parameters and the value of the features
        act_pos = self.net(feats_val)
        act_vel = self.net(feats_dot)
        act = to.cat([act_pos, act_vel], dim=1)

        # Return the flattened tensor if not run in a batch mode to be compatible with the action spaces
        return act.flatten() if not batched else act


class QBallBalancerPDCtrl(Policy):
    """
    PD-controller for the Quanser Ball Balancer.
    The only but significant difference of this controller to the other PD controller is the clipping of the actions.

    .. note::
        This class's desired state specification deviates from the Pyrado policies which interact with a `Task`.
    """

    name: str = 'qbb_pd'

    def __init__(self,
                 env_spec: EnvSpec,
                 state_des: to.Tensor = to.zeros(2),
                 kp: to.Tensor = None,
                 kd: to.Tensor = None,
                 use_cuda: bool = False):
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
        self.kp_servo = 14.  # P-gain for the servo angle; see the saturation block the Simulink model
        self.Kp, self.Kd = None, None
        self.init_param(kp, kd)

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Calculate the controller output.

        :param obs: observation from the environment
        :return act: controller output [V]
        """
        th_x, th_y, x, y, _, _, x_dot, y_dot = obs

        err = to.tensor([self.state_des[0] - x, self.state_des[1] - y])
        err_dot = to.tensor([0. - x_dot, 0. - y_dot])
        th_des = self.Kp.mv(err) + self.Kd.mv(err_dot)

        # Saturation for desired angular position
        th_des = to.clamp(th_des, -self.limit_rad, self.limit_rad)
        err_th = th_des - to.tensor([th_x, th_y])

        # Return action, see "Actuator Electrical Dynamics" block in [1]
        return err_th*self.kp_servo

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
    """ Swing-up and balancing controller for the Quanser Cart-Pole """

    name: str = 'qcp_sub'

    def __init__(self,
                 env_spec: EnvSpec,
                 u_max: float = 18.,
                 v_max: float = 12.,
                 long: bool = False,
                 use_cuda: bool = False):
        """
        Constructor

        :param env_spec: environment specification
        :param u_max: maximum energy gain
        :param v_max: maximum voltage the control signal will be clipped to
        :param long: flag for long or short pole
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(env_spec, use_cuda)

        # Store inputs
        self.u_max = u_max
        self.v_max = v_max
        self.pd_control = False
        self.pd_activated = False
        self.long = long
        self.dp_nom = QCartPoleSim.get_nominal_domain_param(self.long)

        if long:
            self.K_pd = to.tensor([-41.833, 189.8393, -47.8483, 28.0941])
        else:
            self.k_p = to.tensor(8.5)  # former: 8.5
            self.k_d = to.tensor(0.)  # former: 0.
            self.k_e = to.tensor(24.5)  # former: 19.5 (frequency dependent)
            self.K_pd = to.tensor([41., -200., 55., -16.])  # former: [+41.8, -173.4, +46.1, -16.2]

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Calculate the controller output.

        :param obs: observation from the environment
        :return act: controller output [V]
        """
        x, sin_th, cos_th, x_dot, theta_dot = obs
        theta = to.atan2(sin_th, cos_th)
        alpha = (theta - math.pi) if theta > 0 else (theta + math.pi)

        J_pole = self.dp_nom['l_pole']**2*self.dp_nom['m_pole']/3.
        J_eq = self.dp_nom['m_cart'] + (self.dp_nom['eta_g']*self.dp_nom['K_g']**2*
                                        self.dp_nom['J_m'])/self.dp_nom['r_mp']**2

        # Energy terms
        E_kin = J_pole/2.*theta_dot**2
        E_pot = self.dp_nom['m_pole']*self.dp_nom['g']*self.dp_nom['l_pole']*(
            1 - cos_th)  # E(0) = 0., E(pi) = E(-pi) = 2 mgl
        E_ref = 2.*self.dp_nom['m_pole']*self.dp_nom['g']*self.dp_nom['l_pole']

        if to.abs(alpha) < 0.1745 or self.pd_control:
            # Stabilize at the top
            self.pd_activated = True
            u = self.K_pd.dot(to.tensor([x, alpha, x_dot, theta_dot]))
        else:
            # Swing up
            u = self.k_e*(E_kin + E_pot - E_ref)*to.sign(theta_dot*cos_th) + self.k_p*(0. - x) + self.k_d*(0. - x_dot)
            u = u.clamp(-self.u_max, self.u_max)

            if self.pd_activated:
                self.pd_activated = False

        act = (J_eq*self.dp_nom['R_m']*self.dp_nom['r_mp']*u)/ \
              (self.dp_nom['eta_g']*self.dp_nom['K_g']*self.dp_nom['eta_m']*self.dp_nom['k_m']) + \
              self.dp_nom['K_g']*self.dp_nom['k_m']*x_dot/self.dp_nom['r_mp']

        # Return the clipped action
        act = act.clamp(-self.v_max, self.v_max)
        return act.view(1)  # such that when act is later converted to numpy it does not become a float


class QQubeSwingUpAndBalanceCtrl(Policy):
    """ Hybrid controller (QQubeEnergyCtrl, QQubePDCtrl) switching based on the pendulum pole angle alpha

    .. note::
        Extracted Quanser's values from q_qube2_swingup.mdl
    """

    name: str = 'qq_sub'

    def __init__(self,
                 env_spec: EnvSpec,
                 ref_energy: float = 0.025,  # Quanser's value: 0.02
                 energy_gain: float = 50.,  # Quanser's value: 50
                 energy_th_gain: float = 0.4,  # former: 0.4
                 acc_max: float = 5.,  # Quanser's value: 6
                 alpha_max_pd_enable: float = 20.,  # Quanser's value: 20
                 pd_gains: to.Tensor = to.tensor([-2, 35, -1.5, 3]),  # Quanser's value: [-2, 35, -1.5, 3]
                 use_cuda: bool = False):
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

        self.alpha_max_pd_enable = alpha_max_pd_enable/180.*math.pi

        # Set up the energy and PD controller
        self.e_ctrl = QQubeEnergyCtrl(env_spec, ref_energy, energy_gain, energy_th_gain, acc_max)
        self.pd_ctrl = QQubePDCtrl(env_spec, k=pd_gains, al_des=math.pi)

    def pd_enabled(self, cos_al: [float, to.Tensor]) -> bool:
        """
        Check if the PD-controller should be enabled based oin a predefined threshold on the alpha angle.

        :param cos_al: cosine of the pendulum pole angle
        :return: bool if condition is met
        """
        cos_al_delta = 1. + to.cos(to.tensor(math.pi - self.alpha_max_pd_enable))
        return bool(to.abs(1. + cos_al) < cos_al_delta)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.tensor):
        # Reconstruct the sate for the error-based controller
        sin_th, cos_th, sin_al, cos_al, th_d, al_d = obs
        s = to.stack([to.atan2(sin_th, cos_th), to.atan2(sin_al, cos_al), th_d, al_d])

        if self.pd_enabled(cos_al):
            s[1] = s[1]%(2*math.pi)  # alpha can have multiple revolutions
            return self.pd_ctrl(s)
        else:
            return self.e_ctrl(s)


class QQubeEnergyCtrl(Policy):
    """ Energy-based controller used to swing the pendulum up """

    def __init__(self,
                 env_spec: EnvSpec,
                 ref_energy: float,
                 energy_gain: float,
                 th_gain: float,
                 acc_max: float,
                 use_cuda: bool = False):
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

        # Initialize parameters
        self._log_E_ref = nn.Parameter(to.log(to.tensor(ref_energy)), requires_grad=True)
        self._log_E_gain = nn.Parameter(to.log(to.tensor(energy_gain)), requires_grad=True)
        self._th_gain = nn.Parameter(to.tensor(th_gain), requires_grad=True)
        self.acc_max = to.tensor(acc_max)
        self.dp_nom = QQubeSwingUpSim.get_nominal_domain_param()

    @property
    def E_ref(self):
        return to.exp(self._log_E_ref)

    @E_ref.setter
    def E_ref(self, new_E_ref):
        self._log_E_ref = to.log(new_E_ref)

    @property
    def E_gain(self):
        r""" Called $\mu$ by Quanser."""
        return to.exp(self._log_E_gain)

    @E_gain.setter
    def E_gain(self, new_mu):
        r""" Called $\mu$ by Quanser."""
        self._log_E_gain = to.log(new_mu)

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, obs: to.Tensor) -> to.Tensor:
        """
        Control step of energy-based controller which is used in the swing-up controller

        :param obs: observations pre-processed in the `forward` method of `QQubeSwingUpAndBalanceCtrl`
        :return: action
        """
        # Reconstruct partial state
        th, al, thd, ald = obs

        # Compute energies
        J_pole = self.dp_nom['Mp']*self.dp_nom['Lp']**2/12.
        E_kin = 0.5*J_pole*ald**2
        E_pot = 0.5*self.dp_nom['Mp']*self.dp_nom['g']*self.dp_nom['Lp']*(1. - to.cos(al))
        E = E_kin + E_pot

        # Compute clipped action
        u = self.E_gain*(E - self.E_ref)*to.sign(ald*to.cos(al)) - self._th_gain*th
        acc = clamp_symm(u, self.acc_max)
        trq = self.dp_nom['Mr']*self.dp_nom['Lr']*acc
        volt = self.dp_nom['Rm']/self.dp_nom['km']*trq
        return volt.unsqueeze(0)


class QQubePDCtrl(Policy):
    r"""
    PD-controller for the Qunaser Qube.
    Drives Qube to $x_{des} = [\theta_{des}, \alpha_{des}, 0.0, 0.0]$.
    Flag done is set when $|x_des - x| < tol$.
    """

    def __init__(self,
                 env_spec: EnvSpec,
                 k: to.Tensor = to.tensor([4., 0, 0.5, 0]),
                 th_des: float = 0.,
                 al_des: float = 0.,
                 tols: to.Tensor = to.tensor([1.5, 0.5, 0.01, 0.01], dtype=to.float64)/180.*math.pi,
                 use_cuda: bool = False):
        r"""
        Constructor

        :param env_spec: environment specification
        :param k: controller gains, the default values stabilize the pendulum at the center hanging down
        :param th_des: desired rotary pole angle [rad]
        :param al_des: desired pendulum pole angle [rad]
        :param tols: tolerances for the desired angles $\theta$ and $\alpha$ [rad]
        :param use_cuda: `True` to move the policy to the GPU, `False` (default) to use the CPU
        """
        super().__init__(env_spec, use_cuda)

        self.k = nn.Parameter(k, requires_grad=True)
        self.state_des = to.tensor([th_des, al_des, 0., 0.])
        self.tols = tols
        self.done = False

    def init_param(self, init_values: to.Tensor = None, **kwargs):
        pass

    def forward(self, meas: to.Tensor) -> to.Tensor:
        # Unpack the raw measurement (is not an observation)
        err = self.state_des - meas  # th, al, thd, ald

        if all(to.abs(err) <= self.tols):
            self.done = True

        # PD control
        return self.k.dot(err).unsqueeze(0)


class QCartPoleGoToLimCtrl:
    """ Controller for going to one of the joint limits (part of the calibration routine) """

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
        self.u_max = self.sign*np.array([1.5])
        self._t_init = False
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
        if not self._t_init:
            self._t0 = time.time()
            self._t_init = True

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
    """ Controller for going to one of the joint limits (part of the calibration routine) """

    def __init__(self, positive: bool = True, cnt_done: int = 250):
        """
        Constructor

        :param positive: direction switch
        """
        self.done = False
        self.th_lim = 10.
        self.sign = 1 if positive else -1
        self.u_max = 0.8
        self.cnt = 0
        self.cnt_done = cnt_done

    def __call__(self, meas: to.Tensor) -> to.Tensor:
        """
        Go to joint limits by applying `u_max` and save limit value in `th_lim`.

        :param meas: sensor measurement
        :return: action
        """
        # Unpack the raw measurement (is not an observation)
        th = meas[0].item()

        if abs(th - self.th_lim) > 1e-8:
            self.cnt = 0
            self.th_lim = th
        else:
            self.cnt += 1

        # Do this for cnt_done time steps
        self.done = self.cnt >= self.cnt_done
        return to.tensor([self.sign*self.u_max])


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
        if ctrl_type.lower() == 'lqr':
            ctrl_gains = to.tensor([
                [0.1401, 0, 0, 0, -0.09819, -0.1359, 0, 0.545, 0, 0, 0, -0.01417, -0.04427, 0],
                [0, 0.1381, 0, 0.2518, 0, 0, -0.2142, 0, 0.5371, 0, 0.03336, 0, 0, -0.1262],
                [0, 0, 0.1414, 0.0002534, 0, 0, -0.0002152, 0, 0, 0.5318, 0, 0, 0, -0.0001269],
                [0, -0.479, -0.0004812, 39.24, 0, 0, -15.44, 0, -1.988, -0.001934, 9.466, 0, 0, -13.14],
                [0.3039, 0, 0, 0, 25.13, 15.66, 0, 1.284, 0, 0, 0, 7.609, 6.296, 0]
            ])

        elif ctrl_type.lower() == 'h2':
            ctrl_gains = to.tensor([
                [-73.88, -2.318, 39.49, -4.270, 12.25, 0.9779, 0.2564, 35.11, 5.756, 0.8661, -0.9898, 1.421, 3.132,
                 -0.01899],
                [-24.45, 0.7202, -10.58, 2.445, -0.6957, 2.1619, -0.3966, -61.66, -3.254, 5.356, 0.1908, 12.88,
                 6.142, -0.3812],
                [-101.8, -9.011, 64.345, -5.091, 17.83, -2.636, 0.9506, -44.28, 3.206, 37.59, 2.965, -32.65, -21.68,
                 -0.1133],
                [-59.56, 1.56, -0.5794, 26.54, -2.503, 3.827, -7.534, 9.999, 1.143, -16.96, 8.450, -5.302, 4.620,
                 -10.32],
                [-107.1, 0.4359, 19.03, -9.601, 20.33, 10.36, 0.2285, -74.98, -2.136, 7.084, -1.240, 62.62, 33.66,
                 1.790]
            ])

        else:
            raise pyrado.ValueErr(given=ctrl_type, eq_constraint="'lqr' or 'h2'")

        # Compensate for the mismatching different state definition
        if ball_z_dim_mismatch:
            ctrl_gains = insert_tensor_col(ctrl_gains, 7, to.zeros((5, 1)))  # ball z position
            ctrl_gains = insert_tensor_col(ctrl_gains, -1, to.zeros((5, 1)))  # ball z velocity

    elif isinstance(inner_env(env), QBallBalancerSim):
        # Get the controller gains (K-matrix)
        if ctrl_type.lower() == 'pd':
            # Quanser gains (the original Quanser controller includes action clipping)
            ctrl_gains = -to.tensor([[-14., 0, -14*3.45, 0, 0, 0, -14*2.11, 0],
                                     [0, -14., 0, -14*3.45, 0, 0, 0, -14*2.11]])

        elif ctrl_type.lower() == 'lqr':
            # Since the control module can by tricky to install (recommended using anaconda), we only load it if needed
            import control

            # System modeling
            A = np.zeros((env.obs_space.flat_dim, env.obs_space.flat_dim))
            A[:env.obs_space.flat_dim//2, env.obs_space.flat_dim//2:] = np.eye(env.obs_space.flat_dim//2)
            A[4, 4] = -env.B_eq_v/env.J_eq
            A[5, 5] = -env.B_eq_v/env.J_eq
            A[6, 0] = env.c_kin*env.m_ball*env.g*env.r_ball**2/env.zeta
            A[6, 6] = -env.c_kin*env.r_ball**2/env.zeta
            A[7, 1] = env.c_kin*env.m_ball*env.g*env.r_ball**2/env.zeta
            A[7, 7] = -env.c_kin*env.r_ball**2/env.zeta
            B = np.zeros((env.obs_space.flat_dim, env.act_space.flat_dim))
            B[4, 0] = env.A_m/env.J_eq
            B[5, 1] = env.A_m/env.J_eq
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
    ctrl.init_param(-1*ctrl_gains)  # in classical control it is u = -K*x; here a = psi(s)*s
    return ctrl
