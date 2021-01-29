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

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.desired_state import DesStateTask
from pyrado.tasks.reward_functions import AbsErrRewFcn
from pyrado.utils.data_types import RenderMode


class CatapultSim(SimEnv, Serializable):
    """
    In this special environment, the action is equal to the policy parameter. Therefore, it makes only sense to
    use it in combination with a linear policy that has only one constant feature.
    """

    name: str = "cata"

    def __init__(self, max_steps: int, example_config: bool):
        """
        Constructor

        :param max_steps: maximum number of simulation steps
        :param example_config: configuration for the 'illustrative example' in the journal
        """
        Serializable._init(self, locals())

        super().__init__(dt=None, max_steps=max_steps)

        self.example_config = example_config
        self._planet = -1

        # Initialize the domain parameters (Earth)
        self._g = 9.81  # gravity constant [m/s**2]
        self._k = 2e3  # catapult spring's stiffness constant [N/m]
        self._x = 1.0  # catapult spring's pre-elongation [m]

        # Domain independent parameter
        self._m = 70.0  # victim's mass [kg]

        # Set the bounds for the system's states adn actions
        max_state = np.array([1000.0])  # [m], arbitrary but >> self._x
        max_act = max_state
        self._curr_act = np.zeros_like(max_act)  # just for usage in render function

        self._state_space = BoxSpace(-max_state, max_state, labels=["h"])
        self._init_space = SingularStateSpace(np.zeros(self._state_space.shape), labels=["h_0"])
        self._act_space = BoxSpace(-max_act, max_act, labels=["theta"])

        # Define the task including the reward function
        self._task = self._create_task(task_args=dict())

    @property
    def state_space(self):
        return self._state_space

    @property
    def obs_space(self):
        return self._state_space

    @property
    def init_space(self):
        return self._init_space

    @property
    def act_space(self):
        return self._act_space

    def _create_task(self, task_args: dict) -> DesStateTask:
        # Define the task including the reward function
        state_des = task_args.get("state_des", np.zeros(self._state_space.shape))
        return DesStateTask(self.spec, state_des, rew_fcn=AbsErrRewFcn(q=np.array([1.0]), r=np.array([0.0])))

    @property
    def task(self):
        return self._task

    @property
    def domain_param(self) -> dict:
        if self.example_config:
            return dict(planet=self._planet)
        else:
            return dict(g=self._g, k=self._k, x=self._x)

    @domain_param.setter
    def domain_param(self, domain_param: dict):
        assert isinstance(domain_param, dict)
        # Set the new domain params if given, else the default value
        if self.example_config:
            if domain_param["planet"] == 0:
                # Mars
                self._g = 3.71
                self._k = 1e3
                self._x = 0.5
            elif domain_param["planet"] == 1:
                # Venus
                self._g = 8.87
                self._k = 3e3
                self._x = 1.5
            elif domain_param["planet"] == -1:
                # Default value which should make the computation invalid
                self._g = None
                self._k = None
                self._x = None
            else:
                raise pyrado.ValueErr(given=domain_param["planet"], eq_constraint="0 or 1")

        else:
            assert self._g > 0 and self._k > 0 and self._x > 0
            self._g = domain_param.get("g", self._g)
            self._k = domain_param.get("k", self._k)
            self._x = domain_param.get("x", self._x)

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(g=9.81, k=200.0, x=1.0)

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset time
        self._curr_step = 0

        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param

        # Reset the state
        if init_state is None:
            self.state = self._init_space.sample_uniform()  # zero
        else:
            if not init_state.shape == self.obs_space.shape:
                raise pyrado.ShapeErr(given=init_state, expected_match=self.obs_space)
            if isinstance(init_state, np.ndarray):
                self.state = init_state.copy()
            else:
                try:
                    self.state = np.array(init_state)
                except Exception:
                    raise pyrado.TypeErr(given=init_state, expected_type=[np.ndarray, list])

        # Reset the task
        self._task.reset(env_spec=self.spec)

        # Return perfect observation
        return self.observe(self.state)

    def step(self, act):
        # Apply actuator limits
        act = self.limit_act(act)  # dummy for CatapultSim
        self._curr_act = act  # just for the render function

        # Calculate the maximum height of the flight trajectory ("one step dynamics")
        self.state = self._k / (2.0 * self._m * self._g) * (act - self._x) ** 2  # h(theta, xi)

        # Current reward depending on the state after the step (since there is only one step) and the (unlimited) action
        self._curr_rew = self.task.step_rew(self.state, act, self._curr_step)

        self._curr_step += 1

        # Check if the task or the environment is done
        done = self._task.is_done(self.state)
        if self._curr_step >= self._max_steps:
            done = True

        if done:
            # Add final reward if done
            remaining_steps = self._max_steps - (self._curr_step + 1) if self._max_steps is not pyrado.inf else 0
            self._curr_rew += self._task.final_rew(self.state, remaining_steps)

        return self.observe(self.state), self._curr_rew, done, {}

    def render(self, mode: RenderMode, render_step: int = 1):
        # Call base class
        super().render(mode)

        # Print to console
        if mode.text:
            if self._curr_step % render_step == 0 and self._curr_step > 0:  # skip the render before the first step
                print(
                    "step: {:3}  |  r_t: {: 1.3f}  |  a_t: {}\t |  s_t+1: {}".format(
                        self._curr_step, self._curr_rew, self._curr_act, self.state
                    )
                )


class CatapultExample:
    """
    For calculating the quantities of the 'illustrative example' in [1]

    .. seealso::
        [1] F. Muratore, M. Gienger, J. Peters, "Assessing Transferability from Simulation to Reality for Reinforcement
            Learning", PAMI, 2019
    """

    def __init__(self, m, g_M, k_M, x_M, g_V, k_V, x_V):
        """ Constructor """
        # Store parameters
        self.m = m
        self.g_M, self.k_M, self.x_M = g_M, k_M, x_M
        self.g_V, self.k_V, self.x_V = g_V, k_V, x_V

    def opt_policy_param(self, n_M, n_V):
        """
        Compute the optimal policy parameter.

        :param n_M: number of Mars samples
        :param n_V: number of Venus samples
        :return: optimal policy parameter
        """
        # Calculate (mixed-domain) constants
        c_M = n_M * self.k_M * self.g_V
        c_V = n_V * self.k_V * self.g_M

        # Calculate optimal policy parameter
        th_opt = (self.x_M * c_M + self.x_V * c_V) / (c_M + c_V)
        return th_opt

    def opt_est_expec_return(self, n_M, n_V):
        """
        Calculate the optimal objective function value.

        :param n_M: number of Mars samples
        :param n_V: number of Venus samples
        :return: optimal value of the estimated expected return
        """
        c_M = n_M * self.k_M * self.g_V
        c_V = n_V * self.k_V * self.g_M
        c = c_M + c_V

        n = n_M + n_V
        M_part = -n_M * self.k_M / (2 * n * self.m * self.g_M) * ((self.x_V * c_V - self.x_M * c_V) / c) ** 2
        V_part = -n_V * self.k_V / (2 * n * self.m * self.g_V) * ((self.x_M * c_M - self.x_V * c_M) / c) ** 2
        Jhat_n_opt = M_part + V_part

        # Check and return
        assert Jhat_n_opt <= 1e-8, "Jhat_th_n_opt should be <= 0, but was {}!".format(Jhat_n_opt)
        return Jhat_n_opt

    def est_expec_return(self, th, n_M, n_V):
        """
        Calculate the optimal objective function value.

        :param th: policy parameter
        :param n_M: number of Mars samples
        :param n_V: number of Venus samples
        :return: value of the estimated expected return
        """
        n = n_M + n_V
        M_part = -n_M / n * self.k_M / (2 * self.m * self.g_M) * (th - self.x_M) ** 2
        V_part = -n_V / n * self.k_V / (2 * self.m * self.g_V) * (th - self.x_V) ** 2
        Jhat_n = M_part + V_part

        # Check and return
        assert Jhat_n <= 0
        return Jhat_n
