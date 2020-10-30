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

import joblib
import numpy as np
import os.path as osp
import sys
import torch as to
import torch.nn as nn
from copy import deepcopy
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import ReplayMemory
from pyrado.utils.saving_loading import save_prefix_suffix, load_prefix_suffix
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import SACExplStrat
from pyrado.logger.step import StepLogger, ConsolePrinter, CSVPrinter
from pyrado.policies.base import Policy
from pyrado.policies.dummy import RecurrentDummyPolicy, DummyPolicy
from pyrado.policies.two_headed import TwoHeadedPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.utils.input_output import print_cbt, print_cbt_once
from pyrado.utils.data_processing import standardize


class SAC(Algorithm):
    """
    Soft Actor-Critic (SAC) variant with stochastic policy and two Q-functions and two Q-targets (no V-function)

    .. seealso::
        [1] T. Haarnoja, A. Zhou, P. Abbeel, S. Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
        Reinforcement Learning with a Stochastic Actor", ICML, 2018

        [2] This implementation was inspired by https://github.com/pranz24/pytorch-soft-actor-critic
            which is seems to be based on https://github.com/vitchyr/rlkit
    """

    name: str = 'sac'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: TwoHeadedPolicy,
                 q_fcn_1: Policy,
                 q_fcn_2: Policy,
                 memory_size: int,
                 gamma: float,
                 max_iter: int,
                 num_batch_updates: int,
                 tau: float = 0.995,
                 alpha_init: float = 0.2,
                 learn_alpha: bool = True,
                 target_update_intvl: int = 1,
                 standardize_rew: bool = True,
                 batch_size: int = 500,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 num_workers: int = 4,
                 max_grad_norm: float = 5.,
                 lr: float = 3e-4,
                 lr_scheduler=None,
                 lr_scheduler_hparam: [dict, None] = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param q_fcn_1: state-action value function $Q(s,a)$, the associated target Q-functions is created from a
                        re-initialized copies of this one
        :param q_fcn_2: state-action value function $Q(s,a)$, the associated target Q-functions is created from a
                        re-initialized copies of this one
        :param memory_size: number of transitions in the replay memory buffer, e.g. 1000000
        :param gamma: temporal discount factor for the state values
        :param max_iter: number of iterations (policy updates)
        :param num_batch_updates: number of batch updates per algorithm steps
        :param tau: interpolation factor in averaging for target networks, update used for the soft update a.k.a. polyak
                    update, between 0 and 1
        :param alpha_init: initial weighting factor of the entropy term in the loss function
        :param learn_alpha: adapt the weighting factor of the entropy term
        :param target_update_intvl: number of iterations that pass before updating the target network
        :param standardize_rew: bool to flag if the rewards should be standardized
        :param batch_size: number of samples per policy update batch
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_workers: number of environments for parallel sampling
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler type for the policy and the Q-functions that does one step
                             per `update()` call
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if typed_env(env, ActNormWrapper) is None:
            raise pyrado.TypeErr(msg='SAC required an environment wrapped by an ActNormWrapper!')
        if not isinstance(q_fcn_1, Policy):
            raise pyrado.TypeErr(given=q_fcn_1, expected_type=Policy)
        if not isinstance(q_fcn_2, Policy):
            raise pyrado.TypeErr(given=q_fcn_2, expected_type=Policy)

        if logger is None:
            # Create logger that only logs every 100 steps of the algorithm
            logger = StepLogger(print_interval=100)
            logger.printers.append(ConsolePrinter())
            logger.printers.append(CSVPrinter(osp.join(save_dir, 'progress.csv')))

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        # Store the inputs
        self._env = env
        self.q_fcn_1 = q_fcn_1
        self.q_fcn_2 = q_fcn_2
        self.q_targ_1 = deepcopy(self.q_fcn_1)
        self.q_targ_2 = deepcopy(self.q_fcn_2)
        self.q_targ_1.eval()
        self.q_targ_2.eval()
        self.gamma = gamma
        self.tau = tau
        self.learn_alpha = learn_alpha
        self.target_update_intvl = target_update_intvl
        self.standardize_rew = standardize_rew
        self.num_batch_updates = num_batch_updates
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Initialize
        self._memory = ReplayMemory(memory_size)
        if policy.is_recurrent:
            init_expl_policy = RecurrentDummyPolicy(env.spec, policy.hidden_size)
        else:
            init_expl_policy = DummyPolicy(env.spec)
        self.sampler_init = ParallelRolloutSampler(
            self._env, init_expl_policy,  # samples uniformly random from the action space
            num_workers=num_workers,
            min_steps=memory_size,
        )
        self._expl_strat = SACExplStrat(self._policy, std_init=1.)  # std_init will be overwritten by 2nd policy head
        self.sampler = ParallelRolloutSampler(
            self._env, self._expl_strat,
            num_workers=1,
            min_steps=min_steps,  # in [2] this would be 1
            min_rollouts=min_rollouts  # in [2] this would be None
        )
        self.sampler_eval = ParallelRolloutSampler(
            self._env, self._policy,
            num_workers=num_workers,
            min_steps=100*env.max_steps,
            min_rollouts=None
        )
        self._optim_policy = to.optim.Adam([{'params': self._policy.parameters()}], lr=lr, eps=1e-5)
        self._optim_q_fcn_1 = to.optim.Adam([{'params': self.q_fcn_1.parameters()}], lr=lr, eps=1e-5)
        self._optim_q_fcn_2 = to.optim.Adam([{'params': self.q_fcn_2.parameters()}], lr=lr, eps=1e-5)
        log_alpha_init = to.log(to.tensor(alpha_init, dtype=to.get_default_dtype()))
        if learn_alpha:
            # Automatic entropy tuning
            self._log_alpha = nn.Parameter(log_alpha_init, requires_grad=True)
            self._alpha_optim = to.optim.Adam([{'params': self._log_alpha}], lr=lr, eps=1e-5)
            self.target_entropy = -to.prod(to.tensor(env.act_space.shape))
        else:
            self._log_alpha = log_alpha_init

        self._lr_scheduler_policy = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler_policy = lr_scheduler(self._optim_policy, **lr_scheduler_hparam)
            self._lr_scheduler_q_fcn_1 = lr_scheduler(self._optim_q_fcn_1, **lr_scheduler_hparam)
            self._lr_scheduler_q_fcn_2 = lr_scheduler(self._optim_q_fcn_2, **lr_scheduler_hparam)

    @property
    def expl_strat(self) -> SACExplStrat:
        return self._expl_strat

    @property
    def memory(self) -> ReplayMemory:
        """ Get the replay memory. """
        return self._memory

    @property
    def alpha(self) -> to.Tensor:
        """ Get the detached entropy coefficient. """
        return to.exp(self._log_alpha.detach())

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if self._memory.isempty:
            # Warm-up phase
            print_cbt_once('Collecting samples until replay memory contains if full.', 'w')
            # Sample steps and store them in the replay memory
            ros = self.sampler_init.sample()
            self._memory.push(ros)
        else:
            # Sample steps and store them in the replay memory
            ros = self.sampler.sample()
            self._memory.push(ros)

        # Log return-based metrics
        if self._curr_iter%self.logger.print_interval == 0:
            ros = self.sampler_eval.sample()
            rets = [ro.undiscounted_return() for ro in ros]
            ret_max = np.max(rets)
            ret_med = np.median(rets)
            ret_avg = np.mean(rets)
            ret_min = np.min(rets)
            ret_std = np.std(rets)
        else:
            ret_max, ret_med, ret_avg, ret_min, ret_std = 5*[-pyrado.inf]  # dummy values
        self.logger.add_value('max return', np.round(ret_max, 4))
        self.logger.add_value('median return', np.round(ret_med, 4))
        self.logger.add_value('avg return', np.round(ret_avg, 4))
        self.logger.add_value('min return', np.round(ret_min, 4))
        self.logger.add_value('std return', np.round(ret_std, 4))
        self.logger.add_value('avg rollout length', np.round(np.mean([ro.length for ro in ros]), 2))
        self.logger.add_value('num rollouts', len(ros))
        self.logger.add_value('avg memory reward', np.round(self._memory.avg_reward(), 4))

        # Use data in the memory to update the policy and the Q-functions
        self.update()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

    @staticmethod
    def soft_update(target: nn.Module, source: nn.Module, tau: float = 0.995):
        """
        Moving average update, a.k.a. Polyak update.
        Modifies the input argument `target`.

        :param target: PyTroch module with parameters to be updated
        :param source: PyTroch module with parameters to update to
        :param tau: interpolation factor for averaging, between 0 and 1
        """
        if not 0 < tau < 1:
            raise pyrado.ValueErr(given=tau, g_constraint='0', l_constraint='1')

        for targ_param, src_param in zip(target.parameters(), source.parameters()):
            targ_param.data = targ_param.data*tau + src_param.data*(1. - tau)

    def update(self):
        """ Update the policy's and Q-functions' parameters on transitions sampled from the replay memory. """
        # Containers for logging
        policy_losses = to.zeros(self.num_batch_updates)
        expl_strat_stds = to.zeros(self.num_batch_updates)
        q_fcn_1_losses = to.zeros(self.num_batch_updates)
        q_fcn_2_losses = to.zeros(self.num_batch_updates)
        policy_grad_norm = to.zeros(self.num_batch_updates)
        q_fcn_1_grad_norm = to.zeros(self.num_batch_updates)
        q_fcn_2_grad_norm = to.zeros(self.num_batch_updates)

        for b in tqdm(range(self.num_batch_updates), total=self.num_batch_updates,
                      desc=f'Updating', unit='batches', file=sys.stdout, leave=False):

            # Sample steps and the associated next step from the replay memory
            steps, next_steps = self._memory.sample(self.batch_size)
            steps.torch(data_type=to.get_default_dtype())
            next_steps.torch(data_type=to.get_default_dtype())

            # Standardize rewards
            if self.standardize_rew:
                rewards = standardize(steps.rewards).unsqueeze(1)
            else:
                rewards = steps.rewards.unsqueeze(1)
            rew_scale = 1.
            rewards *= rew_scale

            with to.no_grad():
                # Create masks for the non-final observations
                not_done = to.tensor(1. - steps.done, dtype=to.get_default_dtype()).unsqueeze(1)

                # Compute the (next)state-(next)action values Q(s',a') from the target networks
                if self.policy.is_recurrent:
                    next_act_expl, next_log_probs, _ = self._expl_strat(next_steps.observations,
                                                                        next_steps.hidden_states)
                else:
                    next_act_expl, next_log_probs = self._expl_strat(next_steps.observations)
                next_q_val_target_1 = self.q_targ_1(to.cat([next_steps.observations, next_act_expl], dim=1))
                next_q_val_target_2 = self.q_targ_2(to.cat([next_steps.observations, next_act_expl], dim=1))
                next_q_val_target_min = to.min(next_q_val_target_1, next_q_val_target_2) - self.alpha*next_log_probs
                next_q_val = rewards + not_done*self.gamma*next_q_val_target_min

            # Compute the two Q-function losses
            # E_{(s_t, a_t) ~ D} [1/2 * (Q_i(s_t, a_t) - r_t - gamma * E_{s_{t+1} ~ p} [V(s_{t+1})] )^2]
            q_val_1 = self.q_fcn_1(to.cat([steps.observations, steps.actions], dim=1))
            q_val_2 = self.q_fcn_2(to.cat([steps.observations, steps.actions], dim=1))
            q_1_loss = nn.functional.mse_loss(q_val_1, next_q_val)
            q_2_loss = nn.functional.mse_loss(q_val_2, next_q_val)
            q_fcn_1_losses[b] = q_1_loss.data
            q_fcn_2_losses[b] = q_2_loss.data

            # Compute the policy loss
            # E_{s_t ~ D, eps_t ~ N} [log( pi( f(eps_t; s_t) ) ) - Q(s_t, f(eps_t; s_t))]
            if self.policy.is_recurrent:
                act_expl, log_probs, _ = self._expl_strat(steps.observations, steps.hidden_states)
            else:
                act_expl, log_probs = self._expl_strat(steps.observations)
            q1_pi = self.q_fcn_1(to.cat([steps.observations, act_expl], dim=1))
            q2_pi = self.q_fcn_2(to.cat([steps.observations, act_expl], dim=1))
            min_q_pi = to.min(q1_pi, q2_pi)
            policy_loss = to.mean(self.alpha*log_probs - min_q_pi)
            policy_losses[b] = policy_loss.data
            expl_strat_stds[b] = to.mean(self._expl_strat.std.data)

            # Do one optimization step for each optimizer, and clip the gradients if desired
            # Q-fcn 1
            self._optim_q_fcn_1.zero_grad()
            q_1_loss.backward()
            q_fcn_1_grad_norm[b] = self.clip_grad(self.q_fcn_1, None)
            self._optim_q_fcn_1.step()
            # Q-fcn 2
            self._optim_q_fcn_2.zero_grad()
            q_2_loss.backward()
            q_fcn_2_grad_norm[b] = self.clip_grad(self.q_fcn_2, None)
            self._optim_q_fcn_2.step()
            # Policy
            self._optim_policy.zero_grad()
            policy_loss.backward()
            policy_grad_norm[b] = self.clip_grad(self._expl_strat.policy, self.max_grad_norm)
            self._optim_policy.step()

            if self.learn_alpha:
                # Compute entropy coefficient loss
                alpha_loss = -to.mean(self._log_alpha*(log_probs.detach() + self.target_entropy))
                # Do one optimizer step for the entropy coefficient optimizer
                self._alpha_optim.zero_grad()
                alpha_loss.backward()
                self._alpha_optim.step()

            # Soft-update the target networks
            if (self._curr_iter*self.num_batch_updates + b)%self.target_update_intvl == 0:
                SAC.soft_update(self.q_targ_1, self.q_fcn_1, self.tau)
                SAC.soft_update(self.q_targ_2, self.q_fcn_2, self.tau)

        # Update the learning rate if the schedulers have been specified
        if self._lr_scheduler_policy is not None:
            self._lr_scheduler_policy.step()
            self._lr_scheduler_q_fcn_1.step()
            self._lr_scheduler_q_fcn_2.step()

        # Logging
        self.logger.add_value('Q1 loss', to.mean(q_fcn_1_losses).item())
        self.logger.add_value('Q2 loss', to.mean(q_fcn_2_losses).item())
        self.logger.add_value('policy loss', to.mean(policy_losses).item())
        self.logger.add_value('avg policy grad norm', to.mean(policy_grad_norm).item())
        self.logger.add_value('avg expl strat std', to.mean(expl_strat_stds).item())
        self.logger.add_value('alpha', self.alpha.item())
        if self._lr_scheduler_policy is not None:
            self.logger.add_value('learning rate', self._lr_scheduler_policy.get_lr())

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize samplers in case env or policy changed
        self.sampler.reinit(self._env, self._expl_strat)
        self.sampler_eval.reinit(self._env, self._policy)

        # Reset the replay memory
        self._memory.reset()

        # Reset the learning rate schedulers
        if self._lr_scheduler_policy is not None:
            self._lr_scheduler_policy.last_epoch = -1
        if self._lr_scheduler_q_fcn_1 is not None:
            self._lr_scheduler_q_fcn_1.last_epoch = -1
        if self._lr_scheduler_q_fcn_2 is not None:
            self._lr_scheduler_q_fcn_2.last_epoch = -1

    def init_modules(self, warmstart: bool, suffix: str = '', prefix: str = None, **kwargs):
        if prefix is None:
            prefix = f'iter_{self._curr_iter - 1}'

        ppi = kwargs.get('policy_param_init', None)
        t1pi = kwargs.get('target1_param_init', None)
        t2pi = kwargs.get('target2_param_init', None)

        if warmstart and ppi is not None and t1pi is not None and t2pi is not None:
            self._policy.init_param(ppi)
            self.q_targ_1.init_param(t1pi)
            self.q_targ_2.init_param(t2pi)
            print_cbt('Learning given an fixed parameter initialization.', 'w')

        elif warmstart and ppi is None and self._curr_iter > 0:
            self._policy = load_prefix_suffix(
                self._policy, 'policy', 'pt', self.save_dir,
                meta_info=dict(prefix=prefix, suffix=suffix)
            )
            self.q_targ_1 = load_prefix_suffix(
                self.q_targ_1, 'target1', 'pt', self.save_dir,
                meta_info=dict(prefix=prefix, suffix=suffix)
            )
            self.q_targ_2 = load_prefix_suffix(
                self.q_targ_2, 'target2', 'pt', self.save_dir,
                meta_info=dict(prefix=prefix, suffix=suffix)
            )
            print_cbt(f'Learning given the results from iteration {self._curr_iter - 1}', 'w')

        else:
            # Reset the policy
            self._policy.init_param()
            self.q_targ_1.init_param()
            self.q_targ_2.init_param()
            print_cbt('Learning from scratch.', 'w')

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        save_prefix_suffix(self._expl_strat.policy, 'policy', 'pt', self.save_dir, meta_info)
        save_prefix_suffix(self.q_targ_1, 'target1', 'pt', self.save_dir, meta_info)
        save_prefix_suffix(self.q_targ_2, 'target2', 'pt', self.save_dir, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            save_prefix_suffix(self._env, 'env', 'pkl', self.save_dir, meta_info)
