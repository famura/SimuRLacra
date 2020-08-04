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

from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_params import NormalParamNoise, SymmParamExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parameter_exploration_sampler import ParameterSamplingResult
from pyrado.utils.standardizing import standardize


class NES(ParameterExploring):
    """
    Simplified variant of Natural Evolution Strategies (NES)

    .. seealso::
        [1] D. Wierstra, T. Schaul, T. Glasmachers, Y. Sun, J. Peters, J. Schmidhuber, "Natural Evolution Strategies",
        JMLR, 2014

        [2] This implementation was inspired by https://github.com/pybrain/pybrain/blob/master/pybrain/optimization/distributionbased/snes.py
    """

    name: str = 'nes'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 max_iter: int,
                 num_rollouts: int,
                 expl_std_init: float,
                 expl_std_min: float = 0.01,
                 pop_size: int = None,
                 eta_mean: float = 1.,
                 eta_std: float = None,
                 symm_sampling: bool = False,
                 transform_returns: bool = True,
                 num_workers: int = 4,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_rollouts: number of rollouts per policy sample
        :param expl_std_init: initial standard deviation for the exploration strategy
        :param expl_std_min: minimal standard deviation for the exploration strategy
        :param pop_size: number of solutions in the population
        :param eta_mean: step size factor for the mean
        :param eta_std: step size factor for the standard deviation
        :param symm_sampling: use an exploration strategy which samples symmetric populations
        :param transform_returns: use a rank-transformation of the returns to update the policy
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call ParameterExploring's constructor
        super().__init__(
            save_dir,
            env,
            policy,
            max_iter,
            num_rollouts,
            pop_size=pop_size,
            num_workers=num_workers,
            logger=logger
        )

        # Store the inputs
        self.transform_returns = transform_returns

        # Explore using normal noise
        self._expl_strat = NormalParamNoise(
            self._policy.num_param,
            std_init=expl_std_init,
            std_min=expl_std_min,
        )
        if symm_sampling:
            # Exploration strategy based on symmetrical normally distributed noise
            # Symmetric buffer needs to have an even number of samples
            if self.pop_size%2 != 0:
                self.pop_size += 1
            self._expl_strat = SymmParamExplStrat(self._expl_strat)

        # Utility coefficients (ignored for transform_returns = False)
        # Use pop_size + 1 since we are also considering the current policy
        eta_std = eta_std if eta_std is not None else (3 + np.log(policy.num_param))/np.sqrt(self.pop_size + 1)/5.
        self.eta_mean_util, self.eta_std_util = self.compute_utilities(self.pop_size + 1, eta_mean, eta_std)

        # Learning rates [2]
        # Use pop_size + 1 since we are also considering the current policy
        self.lr_mean = 1. if transform_returns else 1e-2
        self.lr_std = 0.6*(3 + np.log(self.pop_size + 1))/3./np.sqrt(self.pop_size + 1)

    @staticmethod
    def compute_utilities(pop_size: int, eta_mean: float, eta_std: float):
        """
        Compute the utilities as described in section 3.1 of [1] (a.k.a. Hansen ranking with uniform baseline)

        :param pop_size: number of solutions in the population
        :param eta_mean: step size factor for the mean
        :param eta_std: step size factor for the standard deviation
        :return: utility coefficient for the mean, and utility coefficient for the standard deviation
        """
        # Compute common utility vector
        log_half = np.log(pop_size/2. + 1)
        log_k = np.log(np.arange(1, pop_size + 1))
        num = np.maximum(0, log_half - log_k)
        utils = num/np.sum(num) - 1./pop_size

        # Convert to PyTorch tensors
        eta_mean_util = to.from_numpy(eta_mean*utils).to(to.get_default_dtype())
        eta_std_util = to.from_numpy(eta_std/2.*utils).to(to.get_default_dtype())
        return eta_mean_util, eta_std_util

    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float = None):
        # Average the return values over the rollouts
        rets_avg_ros = param_results.mean_returns

        # Get the perturbations (deltas from the current policy parameters)
        s = param_results.parameters - self._policy.param_values
        # also divide by the standard deviation to fully standardize
        s /= self._expl_strat.std

        if self.transform_returns:
            # Ascending sort according to return values
            idcs_acs = np.argsort(rets_avg_ros)[::-1]
            s_asc = s[list(idcs_acs), :]

            # Update the mean (see [1, 2])
            delta_mean = self._expl_strat.std*(self.eta_mean_util@s_asc)
            self._policy.param_values += self.lr_mean*delta_mean

            # Update the std (see [1, 2])
            grad_std = self.eta_std_util@(s_asc**2 - 1.)
            new_std = self._expl_strat.std*to.exp(self.lr_std*grad_std/2.)
            self._expl_strat.adapt(std=new_std)

        else:
            # Standardize averaged returns over all pop_size rollouts
            rets_stdized = standardize(rets_avg_ros)
            rets_stdized = to.from_numpy(rets_stdized).to(to.get_default_dtype())

            # delta_mean = 1./len(param_results) * (rets_stdized @ s)
            delta_mean = 1./(self._expl_strat.std*len(param_results))*(rets_stdized@s)
            self._policy.param_values += self.lr_mean*delta_mean

            # Update the std (monotonous exponential decay)
            new_std = self._expl_strat.std*0.999**self._curr_iter
            self._expl_strat.adapt(std=new_std)

        self.logger.add_value('min expl strat std', to.min(self._expl_strat.std))
        self.logger.add_value('avg expl strat std', to.mean(self._expl_strat.std.data).detach().numpy())
        self.logger.add_value('max expl strat std', to.max(self._expl_strat.std))
        self.logger.add_value('expl strat entropy', self._expl_strat.get_entropy().item())
