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
from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union

import pyrado
from pyrado.spaces.base import Space


class RewFcn(ABC):
    """ Base class for all reward functions """

    @abstractmethod
    def __call__(self, s: np.ndarray, a: np.ndarray, remaining_steps: int) -> float:
        """
        Compute the (step) reward.

        :param s: state or state errors (depends on the type of reward function, i.e. subclass)
        :param a: action or action errors (depends on the type of reward function, i.e. subclass)
        :param remaining_steps: number of time steps left in the episode
        :return rew: scalar reward
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """
        Reset internal members. This function is called from the `Task.reset()` function.
        The default implementation does nothing.
        """
        pass


class CompoundRewFcn(RewFcn):
    """ Combine multiple reward functions """

    def __init__(self, rew_fcns: Sequence):
        """
        Constuctor

        :param rew_fcns: sequence, e.g. list or tuple, of reward functions to combine
        """
        if not len(rew_fcns) >= 1:
            raise pyrado.ShapeErr(msg="Provide at least one reward function object!")
        self._fcns = rew_fcns

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray, remaining_steps: int = None) -> float:
        # Return the sum of all individual reward functions (the must operate on the state and action error)
        return sum([fcn(err_s, err_a, remaining_steps) for fcn in self._fcns])

    def reset(self, *args, **kwargs):
        # Pass the reset arguments to all held reward functions
        for fcn in self._fcns:
            fcn.reset(*args, **kwargs)


class ZeroPerStepRewFcn(RewFcn):
    """
    Reward function that yields 0 reward every time step.
    A positive or negative final reward can be specified on the Task-level.
    """

    def __call__(self, err_s: np.ndarray = None, err_a: np.ndarray = None, remaining_steps: int = None) -> float:
        """ None of the inputs matter. """
        return 0.0


class PlusOnePerStepRewFcn(RewFcn):
    """
    Reward function that yields +1 reward every time step.
    A positive or negative final reward can be specified on the Task-level.
    """

    def __call__(self, err_s: np.ndarray = None, err_a: np.ndarray = None, remaining_steps: int = None) -> float:
        """ None of the inputs matter. """
        return 1.0


class MinusOnePerStepRewFcn(RewFcn):
    """
    Reward function that yields -1 reward every time step.
    A positive or negative final reward can be specified on the Task-level.
    """

    def __call__(self, err_s: np.ndarray = None, err_a: np.ndarray = None, remaining_steps: int = None) -> float:
        """ None of the inputs matter. """
        return -1.0


class CosOfOneEleRewFcn(RewFcn):
    """
    Reward function that takes the cosine of one element of the state, given by an index.
    Maximum reward of +1 at state[idx] = +/- state_des[idx], minimum reward of -1 at state[idx] = 0.
    """

    def __init__(self, idx: int):
        """
        Constructor

        :param idx: index of the element (angle) of interest
        """
        self._idx = idx

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray = None, remaining_steps: int = None) -> float:
        return np.cos(err_s[self._idx])


class SSCosOfOneEleRewFcn(CosOfOneEleRewFcn):
    """
    Reward function that takes the shifted and scaled cosine of one element of the state, given by an index.
    Maximum reward of +1 at state[idx] = +/- state_des[idx], minimum reward of 0 at state[idx] = 0.
    """

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray = None, remaining_steps: int = None) -> float:
        # Compute the cos-based reward
        rew_orig = super().__call__(err_s, err_a, remaining_steps)
        # Return the shifted and scaled reward
        return 0.5 * (rew_orig + 1)


class AbsErrRewFcn(RewFcn):
    """ Reward function that returns the negative weighted sum of the absolute errors. """

    def __init__(self, q: np.ndarray, r: np.ndarray):
        """
        Constructor

        :param q: weight vector for the state errors
        :param r: weight vector for the action errors
        """
        if not isinstance(q, np.ndarray) and q.ndim == 1:
            raise pyrado.TypeErr(msg=f"The weights q must be an 1-dim ndarray!")
        if not isinstance(r, np.ndarray) and r.ndim == 1:
            raise pyrado.TypeErr(msg=f"The weights r must be an 1-dim ndarray!")
        if not np.all(q >= 0):
            raise pyrado.ValueErr(given=q, ge_constraint="0")
        if not np.all(r >= 0):
            raise pyrado.ValueErr(given=r, ge_constraint="0")

        self.q = q
        self.r = r

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray, remaining_steps: int = None) -> float:
        # Calculate the reward
        cost = np.abs(err_s).dot(self.q) + np.abs(err_a).dot(self.r)
        return -float(cost)


class QuadrErrRewFcn(RewFcn):
    """ Reward function that returns the exp of the weighted sum of squared errors. """

    def __init__(self, Q: Union[np.ndarray, list], R: Union[np.ndarray, list]):
        """
        Constructor

        :param Q: weight matrix for the state errors (positive semi-definite)
        :param R: weight matrix for the action errors (positive definite)
        """
        if not (isinstance(Q, np.ndarray) and isinstance(R, np.ndarray)):
            try:
                Q = np.array(Q)
                R = np.array(R)
            except Exception:
                raise pyrado.TypeErr(given=Q, expected_type=[np.ndarray, list])
        eig_Q, _ = np.linalg.eig(Q)
        eig_R, _ = np.linalg.eig(R)
        if not (eig_Q >= 0).all():
            raise pyrado.ValueErr(msg=f"The weight matrix Q must not have negative eigenvalues!")
        if not (eig_R >= 0).all():  # in theory strictly > 0
            raise pyrado.ValueErr(msg=f"The weight matrix R must not have negative eigenvalues!")

        self.Q = Q
        self.R = R

    def _weighted_quadr_cost(self, err_s: np.ndarray, err_a: np.ndarray) -> np.ndarray:
        """
        Compute the weighted quadratic cost given state and action error.

        :param err_s: vector of state errors
        :param err_a: vector of action errors (i.e. simply the actions in most cases)
        :return: weighted sum of squared errors
        """
        return err_s.dot(self.Q.dot(err_s)) + err_a.dot(self.R.dot(err_a))

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray, remaining_steps: int = None) -> float:
        assert isinstance(err_s, np.ndarray) and isinstance(err_a, np.ndarray)

        # Adjust shapes (assuming vector shaped state and actions)
        err_s = err_s.reshape(-1)
        err_a = err_a.reshape(-1)

        # Calculate the reward
        cost = self._weighted_quadr_cost(err_s, err_a)  # rew in [-inf, 0]
        return -float(cost)


class ExpQuadrErrRewFcn(QuadrErrRewFcn):
    """ Reward function that returns the exp of the weighted sum of squared errors """

    def __init__(self, Q: Union[np.ndarray, list], R: Union[np.ndarray, list]):
        """
        Constructor

        :param Q: weight matrix for the state errors (positive semi-definite)
        :param R: weight matrix for the action errors (positive definite)
        """
        # Call the constructor of the QuadrErrRewFcn class
        super().__init__(Q, R)

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray, remaining_steps: int = None) -> float:
        # Calculate the cost using the weighted sum of squared errors
        quard_cost = super()._weighted_quadr_cost(err_s, err_a)

        # Calculate the scaled exponential
        rew = np.exp(-quard_cost)  # quard_cost >= 0
        return float(rew)


class ScaledExpQuadrErrRewFcn(QuadrErrRewFcn):
    """ Reward function that returns the exp of the scaled weighted sum of squared errors """

    def __init__(
        self, Q: [np.ndarray, list], R: [np.ndarray, list], state_space: Space, act_space: Space, min_rew: float = 1e-4
    ):
        """
        Constructor
        .. note:: This reward function type depends on environment specific parameters. Due to the domain randomization,
        have to re-init the reward function after every randomization of the env, since obs_max and act_max can change
        when randomizing the domain parameters.

        :param Q: weight matrix for the state errors (positive semi-definite)
        :param R: weight matrix for the action errors (positive definite)
        :param state_space: for extracting the worst case (max cost) observation
        :param act_space: for extracting the worst case (max cost) action
        :param min_rew: minimum reward (only used for the scaling factor in the exponential reward function)
        """
        # Call the constructor of the QuadrErrRewFcn class
        super().__init__(Q, R)

        # Initialize with None
        self.c_max = None

        # Calculate internal members
        self.reset(state_space, act_space, min_rew)

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray, remaining_steps: int = None) -> float:
        # Calculate the cost using the weighted sum of squared errors
        quard_cost = super()._weighted_quadr_cost(err_s, err_a)

        # Calculate the scaled exponential
        rew = np.exp(-self.c_max * quard_cost)  # c_max > 0, quard_cost >= 0
        return float(rew)

    def reset(self, state_space: Space, act_space: Space, min_rew=1e-4, **kwargs):
        if not isinstance(state_space, Space):
            raise pyrado.TypeErr(given=state_space, expected_type=Space)
        if not isinstance(act_space, Space):
            raise pyrado.TypeErr(given=act_space, expected_type=Space)

        # Adjust shapes (assuming vector shaped state and actions)
        state_max = state_space.bound_abs_up.reshape(-1)
        act_max = act_space.bound_abs_up.reshape(-1)

        # Compute the maximum cost for the worst case state-action configuration
        max_cost = state_max.dot(self.Q.dot(state_max)) + act_max.dot(self.R.dot(act_max))
        # Compute a scaling factor that sets the current state and action in relation to the worst case
        self.c_max = -1.0 * np.log(min_rew) / max_cost


class UnderActuatedSwingUpRewFcn(RewFcn):
    """
    Reward function for the swing-up task on the Cart-Pole system similar to [1].

    .. seealso::
        [1] W. Yu, J. Tan, C.K. Liu, G. Turk, "Preparing for the Unknown: Learning a Universal Policy with Online System
            Identification", RSS, 2017
    """

    def __init__(
        self,
        c_pole: float = 1.0,
        c_cart: float = 0.2,
        c_act: float = 1e-3,
        c_theta_sq: float = 1.0,
        c_theta_log: float = 0.1,
        idx_x: int = 0,
        idx_th: int = 1,
    ):
        """
        Constructor

        :param c_pole: scaling parameter for the pole angle cost
        :param c_cart: scaling parameter for the cart position cost
        :param c_act: scaling parameter for the control cost
        :param c_theta_sq: scaling parameter for the quadratic angle deviation
        :param c_theta_log: shifting parameter for the logarithm of the quadratic angle deviation
        :param idx_x: index of he state representing the driving component of the system (e.g. cart position x)
        :param idx_th: index of he state representing the rotating of the system (e.g. pole angle theta)
        """
        self.c_costs = np.array([c_pole, c_cart, c_act])
        self.c_theta_sq = c_theta_sq
        self.c_theta_log = c_theta_log
        self.idx_x = idx_x
        self.idx_th = idx_th

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray, remaining_steps: int = None) -> float:
        cost_pole = self.c_theta_sq * err_s[self.idx_th] ** 2 + np.log(err_s[self.idx_th] ** 2 + self.c_theta_log)
        cost_cart = np.abs(err_s[self.idx_x])  # cart position = cart position error
        cost_act = err_a ** 2
        return float(-self.c_costs @ np.hstack([cost_pole, cost_cart, cost_act]) + 10.0)


class StateBasedRewFcn:
    """
    Reward function which directly operates on the state a.k.a. solution. This class is supposed to be used for wrapping
    classical optimization problems into Pyrado, thus it is negative of the loss function.
    """

    def __init__(self, fcn: Callable[[np.ndarray], float], flip_sign: bool):
        """
        Constructor

        :param fcn: function for evaluating the state a.k.a. solution
        :param flip_sign: return negative of fcn, useful to turn minimization problems into maximization problems
        """
        assert callable(fcn)
        assert isinstance(flip_sign, bool)
        self._fcn = fcn
        self._flip_sign = int(flip_sign)

    @abstractmethod
    def __call__(self, state: np.ndarray) -> float:
        """
        Compute the (step) reward.

        :param state: state
        :return: scalar reward
        """
        return -(1 ** self._flip_sign) * self._fcn(state)


class ForwardVelocityRewFcn(RewFcn):
    """
    Reward function for the `HalfCheetahSim` and `SwimmerSim` environment, encouraging to run forward

    .. note::
        The OpenAi Gym calculates the velocity via forward differences, while here we get the velocity directly from
        the simulator.

    .. seealso::
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/half_cheetah.py
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/swimmer_v3.py
    """

    def __init__(self, dt: float, idx_fwd: int, fwd_rew_weight: float, ctrl_cost_weight: float):
        """
        Constructor

        .. note::
            The last x position, which is rewarded, is initialized by `reset()`, since the (sampled) initial state is
            unknown at construction time of the task, i.e. this reward function.

        :param dt: simulation step size [s]
        :param idx_fwd: index of the state dimension that marks the forward direction
        :param fwd_rew_weight: scaling factor for the forward velocity reward
        :param ctrl_cost_weight: scaling factor for the control cost
        """
        self._dt = dt
        self._idx_x_pos = idx_fwd
        self.last_x_pos = None
        self.fwd_rew_weight = fwd_rew_weight
        self.ctrl_cost_weight = ctrl_cost_weight

    def reset(self, init_state, **kwargs):
        self.last_x_pos = init_state[self._idx_x_pos]

    def __call__(self, state: np.ndarray, act: np.ndarray, remaining_steps: int = None) -> float:
        # Operate on the state and actions
        fwd_vel_rew = self.fwd_rew_weight * (state[self._idx_x_pos] - self.last_x_pos) / self._dt
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(act))

        self.last_x_pos = state[self._idx_x_pos]

        return float(fwd_vel_rew - ctrl_cost)


class QCartPoleSwingUpCustomRewFcn(RewFcn):
    """ Custom reward function for QCartPoleSwingUpSim. """

    def __init__(self, factor=0.9):
        """
        Constructor
        :param factor: weighting factor of rotation error to position error
        """
        self.factor = factor

    def __call__(self, err_s: np.ndarray, err_a: np.ndarray, remaining_steps: int = None) -> float:
        assert isinstance(err_s, np.ndarray) and isinstance(err_a, np.ndarray)

        # Reward should be roughly between [0, 1]
        return self.factor*(1-np.abs(err_s[1]/np.pi)**2) + (1-self.factor)*(np.abs(err_s[0])) 
