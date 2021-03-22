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
import torch as to
from copy import deepcopy
from init_args_serializer import Serializable
from torch.distributions import MultivariateNormal

import pyrado
from pyrado.environments.sim_base import SimEnv
from pyrado.spaces import BoxSpace
from pyrado.spaces.base import Space
from pyrado.spaces.empty import EmptySpace
from pyrado.spaces.singular import SingularStateSpace
from pyrado.tasks.goalless import OptimProxyTask
from pyrado.tasks.reward_functions import StateBasedRewFcn
from pyrado.utils.data_types import RenderMode


class TwoDimGaussian(SimEnv, Serializable):
    """
    A toy model with complex 2-dim Gaussian posterior as described in [1].
    We use the domain parameters to capture the

    .. seealso::
        [1] G. Papamakarios, D. Sterratt, I. Murray, "Sequential Neural Likelihood: Fast Likelihood-free Inference with
            Autoregressive Flows", AISTATS, 2019
    """

    name: str = "2dg"

    def __init__(self):
        """ Constructor """
        Serializable._init(self, locals())

        # Call SimEnv's constructor
        super().__init__(dt=1.0, max_steps=1)

        # Initialize the domain parameters and the derived constants
        self._domain_param = self.get_nominal_domain_param()
        self._mean, self._covariance_matrix = self.calc_constants(self.domain_param)

        self._init_space = SingularStateSpace(np.zeros(self.state_space.shape))

        # Define a dummy task including the reward function
        self._task = self._create_task()

    def _to_scalar(self):
        for param in self._domain_param:
            if isinstance(self._domain_param[param], to.Tensor):  # or isinstance(self._domain_param, np.ndarray):
                self._domain_param[param] = self._domain_param[param].item()

    @staticmethod
    def calc_constants(dp):
        for param in dp:
            if isinstance(dp[param], to.Tensor):  # or isinstance(self._domain_param, np.ndarray):
                dp[param] = dp[param].item()
        mean = np.array([dp["m_1"], dp["m_2"]])
        s1 = dp["s_1"] ** 2
        s2 = dp["s_2"] ** 2
        rho = np.tanh(dp["rho"])
        cov12 = rho * s1 * s2
        covariance_matrix = np.array([[s1 ** 2, cov12], [cov12, s2 ** 2]]) + 1e-6 * np.eye(2)
        return mean, covariance_matrix

    @property
    def constants(self):
        return self._mean, self._covariance_matrix

    @property
    def state_space(self):
        max_state = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        return BoxSpace(-max_state, max_state, labels=["x_1", "x_2", "x_2", "x_3", "x_5", "x_6", "x_7", "x_8"])

    @property
    def obs_space(self):
        return self.state_space

    @property
    def init_space(self):
        return self._init_space

    @init_space.setter
    def init_space(self, space: Space):
        if not isinstance(space, Space):
            raise pyrado.TypeErr(given=space, expected_type=Space)
        self._init_space = space

    @property
    def act_space(self):
        return EmptySpace()

    def _create_task(self, task_args: dict = None) -> OptimProxyTask:
        # Dummy task
        return OptimProxyTask(self.spec, StateBasedRewFcn(lambda x: 0.0))

    @property
    def task(self) -> OptimProxyTask:
        return self._task

    @property
    def domain_param(self) -> dict:
        return deepcopy(self._domain_param)

    @domain_param.setter
    def domain_param(self, param: dict):
        if not isinstance(param, dict):
            raise pyrado.TypeErr(given=param, expected_type=dict)
        # Update the parameters
        self._domain_param.update(param)

    @classmethod
    def get_nominal_domain_param(cls) -> dict:
        return dict(
            m_1=0.7,  # first mean
            m_2=-1.5,  # second mean
            s_1=-1,  # first std, also used for coupling term
            s_2=-0.9,  # second std, also used for coupling term
            rho=0.6,  # scaling factor
        )

    def reset(self, init_state: np.ndarray = None, domain_param: dict = None) -> np.ndarray:
        # Reset the domain parameters
        if domain_param is not None:
            self.domain_param = domain_param
            self._mean, self._covariance_matrix = self.calc_constants(self.domain_param)

        # Reset the state
        if init_state is None:
            # Sample from the init state space
            self.state = self.init_space.sample_uniform()
        else:
            if not init_state.shape == self.obs_space.shape:
                raise pyrado.ShapeErr(given=init_state, expected_match=self.obs_space)
            if isinstance(init_state, np.ndarray):
                self.state = init_state.copy()
            else:
                try:
                    self.state = np.asarray(init_state)
                except Exception:
                    raise pyrado.TypeErr(given=init_state, expected_type=np.ndarray)

        # Reset time
        self._curr_step = 0

        # No need to reset the task

        # Return perfect observation
        return self.observe(self.state)

    def step(self, act: np.ndarray = None) -> tuple:
        # Draw 4 samples from the 2-dim complex posterior
        self.state = np.random.multivariate_normal(self._mean, self._covariance_matrix, size=4).flatten()

        # Current reward depending on the state after the step (since there is only one step)
        self._curr_rew = self.task.step_rew(self.state)

        self._curr_step += 1

        # Check if the task or the environment is done
        done = False
        if self._curr_step == self.max_steps:
            done = True

        return self.observe(self.state), self._curr_rew, done, {}

    def render(self, mode: RenderMode, render_step: int = 1):
        # Call base class
        super().render(mode)

        # Print to console
        if mode.text:
            if self._curr_step % render_step == 0 and self._curr_step > 0:  # skip the render before the first step
                print(f"step: {self._curr_step:4d}  |  r_t: {self._curr_rew: 1.3f}  |  s_t+1: {self.state}")

    @staticmethod
    def log_prob(trajectory, params):
        """
        Very ugly, but can be used to calculate the probability of a rollout in the case that we are interested on
        the exact posterior probability

        Calculates the log-probability for a pair of states and domain parameters.
        """
        # TODO: check if params has batch size
        params_names = list(TwoDimGaussian.get_nominal_domain_param().keys())
        log_probs = []
        for param in params:
            mean, covariance_matrix = TwoDimGaussian.calc_constants(dict(zip(params_names, param)))
            dist = MultivariateNormal(loc=to.tensor(mean), covariance_matrix=to.tensor(covariance_matrix))
            log_prob = to.zeros((1,))
            len_traj = len(trajectory) // 2
            for i in range(len_traj):
                log_prob += dist.log_prob(trajectory[[i, i + 1]])
            log_probs.append(log_prob)
        log_probs = to.stack(log_probs)
        return log_probs
