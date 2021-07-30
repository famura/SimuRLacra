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
import copy
from typing import Optional

import torch as to
import torch.nn as nn

import pyrado
from pyrado.algorithms.episodic.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_params import EnergyParamNoise
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.feed_back.fnn import FNN
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult


class EnergyREPS(ParameterExploring):
    """
    Energy-based episodic REPS (eneREPS)

    Instead of fitting a Gaussian at every episode, eneREPS samples from the complete posterior where the energy is
    estimated by a fitted neural network.
    """

    name: str = "enereps"

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env: Env,
        policy: Policy,
        max_iter: int,
        pop_size: Optional[int],
        num_init_states_per_domain: int,
        num_domains: int = 1,
        num_iter_energy: int = 1000,
        lr_energy: float = 1e-2,
        num_workers: int = 4,
        logger: Optional[StepLogger] = None,
    ):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param pop_size: number of solutions in the population
        :param num_init_states_per_domain: number of rollouts to cover the variance over initial states
        :param num_domains: number of rollouts due to the variance over domain parameters
        :param num_iter_energy: number of iterations for the energy-based exploration noise update
        :param lr_energy: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call ParameterExploring's constructor
        super().__init__(
            save_dir=save_dir,
            env=env,
            policy=policy,
            max_iter=max_iter,
            num_init_states_per_domain=num_init_states_per_domain,
            num_domains=num_domains,
            pop_size=pop_size,
            num_workers=num_workers,
            logger=logger,
        )

        # Explore using an energy-based distribution
        self._expl_strat = EnergyParamNoise(
            param_dim=self._policy.num_param,
            energy_net=FNN(
                input_size=self._policy.num_param, output_size=1, hidden_sizes=[64, 64], hidden_nonlin=to.tanh
            ),
            num_iter=num_iter_energy,
            lr=lr_energy,
            use_cuda=self._policy.device != "cpu",
        )

    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float = None):
        # Average the return values over the rollouts
        rets_avg_ros = to.from_numpy(param_results.mean_returns).to(to.get_default_dtype())

        old_energy_net = copy.deepcopy(self._expl_strat.noise.net)

        # Update the exploration strategy
        self._expl_strat.adapt(params=param_results.parameters.detach(), values=rets_avg_ros)

        # Set the prior to the previous net
        self._expl_strat.noise.prior = old_energy_net

        # TODO Update the policy based on the likelihood of drawn samples and the prior
        samples = self._expl_strat.sample_param_sets(
            self._policy.param_values, self.pop_size
        )  # sampling around the current policy

        # Estimate the log-prob of the sampled particles with the updated net
        log_porbs = self._expl_strat.noise.log_prob(samples)  # batched eval via nn.Module
        idx_argmax = to.argmax(log_porbs)  # highest mode
        self._policy.param_values = samples[idx_argmax, :]  # new (intermediate) best policy parameter set

        # Logging
        # kl_e = kl_divergence(distr_new, distr_old)  # mode seeking a.k.a. exclusive KL
        # kl_i = kl_divergence(distr_old, distr_new)  # mean seeking a.k.a. inclusive KL
        # self.logger.add_value("KL(new_old)", kl_e, 6)
        # self.logger.add_value("KL(old_new)", kl_i, 6)
