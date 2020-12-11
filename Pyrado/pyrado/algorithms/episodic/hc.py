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
from typing import Optional

import pyrado
from pyrado.algorithms.episodic.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.utils.input_output import print_cbt
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult
from pyrado.exploration.stochastic_params import HyperSphereParamNoise, NormalParamNoise
from abc import abstractmethod


class HC(ParameterExploring):
    """
    Hill Climbing (HC)

    HC is a heuristic-based policy search method that samples a population of policy parameters per iteration
    and evaluates them on multiple rollouts. If one of the new parameters is better than the current one it is kept.
    If the exploration parameters grow too large, they are reset.
    """

    name: str = "hc"

    def __init__(
        self,
        save_dir: str,
        env: Env,
        policy: Policy,
        max_iter: int,
        num_rollouts: int,
        expl_factor: float,
        pop_size: Optional[int] = None,
        num_workers: int = 4,
        logger: Optional[StepLogger] = None,
    ):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_rollouts: number of rollouts per policy sample
        :param expl_factor: scalar value which determines how the exploration strategy adapts its search space
        :param pop_size: number of solutions in the population
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call ParameterExploring's constructor
        super().__init__(
            save_dir, env, policy, max_iter, num_rollouts, pop_size=pop_size, num_workers=num_workers, logger=logger
        )

        # Store the inputs
        self.expl_factor = float(expl_factor)

        # Parameters for reset heuristics
        self.max_policy_param = 1e3

    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float):
        # Average the return values over the rollouts
        rets_avg_ros = param_results.mean_returns

        # Update the policy
        if np.max(rets_avg_ros) > ret_avg_curr:
            # Update the policy parameters to the best solution from the current population
            idx_max = np.argmax(rets_avg_ros)
            self._policy.param_values = param_results[idx_max].params

        # Re-initialize the policy parameters if the became too large
        if (to.abs(self._policy.param_values) > self.max_policy_param).any():
            self._policy.init_param()
            print_cbt("Reset policy parameters.", "y")

        # Update exploration strategy in subclass
        self.update_expl_strat(rets_avg_ros, ret_avg_curr)

    @abstractmethod
    def update_expl_strat(self, rets_avg_ros: np.ndarray, ret_avg_curr: float):
        raise NotImplementedError


class HCNormal(HC):
    """ Hill Climbing variant using an exploration strategy with normally distributed noise on the policy parameters """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        :param expl_std_init: initial standard deviation for the exploration strategy
        :param args: forwarded the superclass constructor
        :param kwargs: forwarded the superclass constructor
        """
        # Preprocess inputs and call HC's constructor
        expl_std_init = kwargs.pop("expl_std_init")
        if "expl_r_init" in kwargs:
            # This is just for the ability to create one common hyper-param list for HCNormal and HCHyper
            kwargs.pop("expl_r_init")

        # Get from kwargs with default values
        expl_std_min = kwargs.pop("expl_std_min", 0.01)

        # Call HC's constructor
        super().__init__(*args, **kwargs)

        self._expl_strat = NormalParamNoise(
            param_dim=self._policy.num_param,
            std_init=expl_std_init,
            std_min=expl_std_min,
        )

    def update_expl_strat(self, rets_avg_ros: np.ndarray, ret_avg_curr: float):
        # Update the exploration distribution
        if np.max(rets_avg_ros) > ret_avg_curr:
            self._expl_strat.adapt(std=self._expl_strat.std / self.expl_factor ** 2)
        else:
            self._expl_strat.adapt(std=self._expl_strat.std * self.expl_factor)

        self.logger.add_value("min expl strat std", to.min(self._expl_strat.std), 4)
        self.logger.add_value("avg expl strat std", to.mean(self._expl_strat.std), 4)
        self.logger.add_value("max expl strat std", to.max(self._expl_strat.std), 4)
        self.logger.add_value("expl strat entropy", self._expl_strat.get_entropy(), 4)


class HCHyper(HC):
    """ Hill Climbing variant using an exploration strategy that samples policy parameters from a hyper-sphere """

    def __init__(self, *args, **kwargs):
        """
        Constructor

        :param expl_r_init: initial radius of the hyper sphere for the exploration strategy
        :param args: forwarded the superclass constructor
        :param kwargs: forwarded the superclass constructor
        """
        # Preprocess inputs and call HC's constructor
        expl_r_init = kwargs.pop("expl_r_init")
        if expl_r_init <= 0:
            raise pyrado.ValueErr(given=expl_r_init, g_constraint="0")

        if "expl_std_init" in kwargs:
            # This is just for the ability to create one common hyper-param list for HCNormal and HCHyper
            kwargs.pop("expl_std_init")

        # Get from kwargs with default values
        self.expl_r_min = kwargs.pop("expl_r_min", 0.01)
        self.expl_r_max = max(expl_r_init, kwargs.pop("expl_r_max", 10.0))

        # Call HC's constructor
        super().__init__(*args, **kwargs)

        self._expl_strat = HyperSphereParamNoise(
            param_dim=self._policy.num_param,
            expl_r_init=expl_r_init,
        )

    def update_expl_strat(self, rets_avg_ros: np.ndarray, ret_avg_curr: float):
        # Update the exploration strategy
        if np.max(rets_avg_ros) > ret_avg_curr:
            self._expl_strat.adapt(r=self._expl_strat.r / self.expl_factor ** 2)
        else:
            self._expl_strat.adapt(r=self._expl_strat.r * self.expl_factor)

        # Re-initialize the exploration parameters if the became too small or too large
        if self._expl_strat.r < self.expl_r_min or self._expl_strat.r > self.expl_r_max:
            self._expl_strat.reset_expl_params()

        self.logger.add_value("smallest expl param", self._expl_strat.r, 4)
        self.logger.add_value("largest expl param", self._expl_strat.r, 4)
