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

import torch as to
import numpy as np
import joblib
import os.path as osp
from abc import abstractmethod
from copy import deepcopy

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parameter_exploration_sampler import ParameterExplorationSampler, ParameterSamplingResult
from pyrado.exploration.stochastic_params import StochasticParamExplStrat
from pyrado.utils.input_output import print_cbt


class ParameterExploring(Algorithm):
    """ Base for all algorithms that explore directly in the policy parameter space """

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 max_iter: int,
                 num_rollouts: int,
                 pop_size: [int, None] = None,
                 num_sampler_envs: int = 4,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_rollouts: number of rollouts per policy parameter set
        :param pop_size: number of solutions in the population, pass `None` to use a default that scales logarithmically
                         with the number of policy parameters
        :param num_sampler_envs: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not (isinstance(pop_size, int) or pop_size is None):
            raise pyrado.TypeErr(given=pop_size, expected_type=int)
        if isinstance(pop_size, int) and pop_size <= 0:
            raise pyrado.ValueErr(given=pop_size, g_constraint='0')

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        # Store the inputs
        self._env = env
        self.num_rollouts = num_rollouts

        # Auto-select population size if needed
        if pop_size is None:
            pop_size = 4 + int(3*np.log(policy.num_param))
            print_cbt(f'Initialized population size to {pop_size}.', 'y')
        self.pop_size = pop_size

        # Create sampler
        self.sampler = ParameterExplorationSampler(
            env,
            policy,
            num_envs=num_sampler_envs,
            num_rollouts_per_param=num_rollouts,
        )

        # Stopping criterion
        self.ret_avg_stack = 1e3*np.random.randn(20)  # stack size = 20
        self.thold_ret_std = 1e-1  # algorithm terminates if below for multiple iterations

        # Saving the best policy (this is not the mean for policy parameter exploration)
        self.best_policy_param = policy.param_values.clone()

        # Set this in subclasses
        self._expl_strat = None

    @property
    def env(self) -> Env:
        """ Get the environment in which the algorithm exploration trains. """
        return self._env

    @property
    def expl_strat(self) -> StochasticParamExplStrat:
        return self._expl_strat

    def stopping_criterion_met(self) -> bool:
        """
        Check if the average reward of the mean policy did not change more than the specified threshold over the
        last iterations.
        """
        if np.std(self.ret_avg_stack) < self.thold_ret_std:
            return True
        else:
            return False

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Reset the best policy parameters
        self.best_policy_param = self._policy.param_values.clone()

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Sample new policy parameters
        param_sets = self._expl_strat.sample_param_sets(
            self._policy.param_values,
            self.pop_size,
            # If you do not want to include the current policy parameters, be aware that you also have to do follow-up
            # changes in the update() functions in all subclasses of ParameterExploring
            include_nominal_params=True
        )

        with to.no_grad():
            # Sample rollouts using these parameters
            param_samp_res = self.sampler.sample(param_sets)

        # Evaluate the current policy (first one in list if include_nominal_params is True)
        ret_avg_curr = param_samp_res[0].mean_undiscounted_return

        # Store the average return for the stopping criterion
        self.ret_avg_stack = np.delete(self.ret_avg_stack, 0)
        self.ret_avg_stack = np.append(self.ret_avg_stack, ret_avg_curr)

        all_rets = param_samp_res.mean_returns
        all_lengths = np.array([len(ro) for pss in param_samp_res for ro in pss.rollouts])

        # Log metrics computed from the old policy (before the update)
        self.logger.add_value('curr policy return', ret_avg_curr)
        self.logger.add_value('max return', float(np.max(all_rets)))
        self.logger.add_value('median return', float(np.median(all_rets)))
        self.logger.add_value('min return', float(np.min(all_rets)))
        self.logger.add_value('avg return', float(np.mean(all_rets)))
        self.logger.add_value('std return', float(np.std(all_rets)))
        self.logger.add_value('avg rollout len', float(np.mean(all_lengths)))
        self.logger.add_value('min mag policy param',
                              self._policy.param_values[to.argmin(abs(self._policy.param_values))])
        self.logger.add_value('max mag policy param',
                              self._policy.param_values[to.argmax(abs(self._policy.param_values))])

        # Extract the best policy parameter sample for saving it later
        self.best_policy_param = param_samp_res.parameters[np.argmax(param_samp_res.mean_returns)].clone()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(np.max(param_samp_res.mean_returns)), meta_info)

        # Update the policy
        self.update(param_samp_res, ret_avg_curr)

    @abstractmethod
    def update(self, param_results: ParameterSamplingResult, ret_avg_curr: float):
        """
        Update the policy from the given samples.

        :param param_results: Sampled parameters with evaluation
        :param ret_avg_curr: Average return for the current parameters
        """
        raise NotImplementedError

    def save_snapshot(self, meta_info: dict = None):
        # Algorithm.save_snapshot() saves the policy used for exploring, we override it here with the best policy
        super().save_snapshot(meta_info)

        best_policy = deepcopy(self._policy)
        best_policy.param_values = self.best_policy_param

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env, osp.join(self._save_dir, 'env.pkl'))
            to.save(best_policy, osp.join(self._save_dir, 'policy.pt'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            if 'prefix' in meta_info and 'suffix' in meta_info:
                to.save(best_policy, osp.join(self._save_dir,
                                              f"{meta_info['prefix']}_policy_{meta_info['suffix']}.pt"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                to.save(best_policy, osp.join(self._save_dir, f"{meta_info['prefix']}_policy.pt"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                to.save(best_policy, osp.join(self._save_dir, f"policy_{meta_info['suffix']}.pt"))
            else:
                raise NotImplementedError

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir
        super().load_snapshot(ld, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._env = joblib.load(osp.join(ld, 'env.pkl'))
        else:
            # This algorithm instance is a subroutine of a meta-algorithm
            pass
