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
import sys
import torch as to
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
from typing import Sequence

import pyrado
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.utils import compute_action_statistics, num_iter_from_rollouts
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.base_recurrent import RecurrentPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.step_sequence import StepSequence, discounted_values
from pyrado.utils.math import explained_var


class PPO(ActorCritic):
    """
    Proximal Policy Optimization (PPO)

    .. seealso::
        [1] J. Schulmann,  F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms",
        arXiv, 2017

        [2] D.P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", ICLR, 2015
    """

    name: str = 'ppo'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 critic: GAE,
                 max_iter: int,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 num_epoch: int = 3,
                 eps_clip: float = 0.1,
                 batch_size: int = 64,
                 std_init: float = 1.0,
                 num_workers: int = 4,
                 max_grad_norm: float = None,
                 lr: float = 5e-4,
                 lr_scheduler=None,
                 lr_scheduler_hparam: [dict, None] = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param critic: advantage estimation function $A(s,a) = Q(s,a) - V(s)$
        :param max_iter: number of iterations (policy updates)
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_epoch: number of iterations over all gathered samples during one policy update
        :param eps_clip: max/min probability ratio, see [1]
        :param batch_size: number of samples per policy update batch
        :param std_init: initial standard deviation on the actions for the exploration noise
        :param num_workers: number of environments for parallel sampling
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created

        .. note::
            The Adam optimizer computes individual learning rates for all parameters. Thus, the learning rate scheduler
            schedules the maximum learning rate.
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        assert isinstance(policy, Policy)

        # Call ActorCritic's constructor
        super().__init__(env, policy, critic, save_dir, max_iter, logger)

        # Store the inputs
        self.num_epoch = num_epoch
        self.eps_clip = eps_clip
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Initialize
        self.log_loss = True
        self._expl_strat = NormalActNoiseExplStrat(self._policy, std_init=std_init)
        self.sampler = ParallelRolloutSampler(
            env, self._expl_strat,
            num_workers=num_workers,
            min_steps=min_steps,
            min_rollouts=min_rollouts
        )
        self.optim = to.optim.Adam(
            [{'params': self._expl_strat.policy.parameters()},
             {'params': self._expl_strat.noise.parameters()}],
            lr=lr, eps=1e-5
        )
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self._expl_strat

    def loss_fcn(self, log_probs: to.Tensor, log_probs_old: to.Tensor, adv: to.Tensor) -> to.Tensor:
        """
        PPO loss function

        :param log_probs: logarithm of the probabilities of the taken actions using the updated policy
        :param log_probs_old: logarithm of the probabilities of the taken actions using the old policy
        :param adv: advantage values
        :return: loss value
        """
        prob_ratio = to.exp(log_probs - log_probs_old)
        pr_clip = prob_ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip)
        return -to.mean(to.min(prob_ratio*adv.to(self.policy.device), pr_clip*adv.to(self.policy.device)))

    def update(self, rollouts: Sequence[StepSequence]):
        # Turn the batch of rollouts into a list of steps
        concat_ros = StepSequence.concat(rollouts)
        concat_ros.torch(data_type=to.get_default_dtype())

        # Update the advantage estimator's parameters and return advantage estimates
        adv = self._critic.update(rollouts, use_empirical_returns=False)

        with to.no_grad():
            # Compute the action probabilities using the old (before update) policy
            act_stats = compute_action_statistics(concat_ros, self._expl_strat)
            log_probs_old = act_stats.log_probs
            act_distr_old = act_stats.act_distr

        # Attach advantages and old log probs to rollout
        concat_ros.add_data('adv', adv)
        concat_ros.add_data('log_probs_old', log_probs_old)

        # For logging the gradient norms
        policy_grad_norm = []

        # Iterations over the whole data set
        for e in range(self.num_epoch):

            for batch in tqdm(
                concat_ros.split_shuffled_batches(self.batch_size, complete_rollouts=self._policy.is_recurrent),
                total=num_iter_from_rollouts(None, concat_ros, self.batch_size),
                desc=f'Epoch {e}', unit='batches', file=sys.stdout, leave=False):
                # Reset the gradients
                self.optim.zero_grad()

                # Compute log of the action probabilities for the mini-batch
                log_probs = compute_action_statistics(batch, self._expl_strat).log_probs.to(self.policy.device)

                # Compute policy loss and backpropagate
                loss = self.loss_fcn(log_probs, batch.log_probs_old.to(self.policy.device), batch.adv.to(self.policy.device))
                loss.backward()

                # Clip the gradients if desired
                policy_grad_norm.append(self.clip_grad(self._expl_strat.policy, self.max_grad_norm))

                # Call optimizer
                self.optim.step()

                if to.isnan(self._expl_strat.noise.std).any():
                    raise RuntimeError(f'At least one exploration parameter became NaN! The exploration parameters are'
                                       f'\n{self._expl_strat.std.detach().cpu().numpy()}')

            # Update the learning rate if a scheduler has been specified
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

        # Additional logging
        if self.log_loss:
            with to.no_grad():
                act_stats = compute_action_statistics(concat_ros, self._expl_strat)
                log_probs_new = act_stats.log_probs
                act_distr_new = act_stats.act_distr
                loss_after = self.loss_fcn(log_probs_new, log_probs_old, adv)
                kl_avg = to.mean(kl_divergence(act_distr_old, act_distr_new))  # mean seeking a.k.a. inclusive KL
                self.logger.add_value('loss after', loss_after.detach().cpu().numpy())
                self.logger.add_value('KL(old_new)', kl_avg.item())

        # Logging
        self.logger.add_value('avg expl strat std', to.mean(self._expl_strat.noise.std.data).detach().cpu().numpy())
        self.logger.add_value('expl strat entropy', self._expl_strat.noise.get_entropy().item())
        self.logger.add_value('avg policy grad norm', np.mean(policy_grad_norm))
        if self._lr_scheduler is not None:
            self.logger.add_value('learning rate', self._lr_scheduler.get_lr())


class PPO2(ActorCritic):
    """
    Variant of Proximal Policy Optimization (PPO)
    PPO2 differs from PPO by also clipping the value function and standardizing the advantages.

    .. note::
        PPO2 refers to the OpenAI version of PPO which is a GPU-enabled implementation. However, this one is not!

    .. seealso::
        [1] OpenAI Stable Baselines Documentation, https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

        [2] J. Schulmann,  F. Wolski, P. Dhariwal, A. Radford, O. Klimov, "Proximal Policy Optimization Algorithms",
        arXiv, 2017

        [3] D.P. Kingma, J. Ba, "Adam: A Method for Stochastic Optimization", ICLR, 2015
    """

    name: str = 'ppo2'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 critic: GAE,
                 max_iter: int,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 num_epoch: int = 3,
                 eps_clip: float = 0.1,
                 value_fcn_coeff: float = 0.5,
                 entropy_coeff: float = 1e-3,
                 batch_size: int = 32,
                 std_init: float = 1.0,
                 num_workers: int = 4,
                 max_grad_norm: float = None,
                 lr: float = 5e-4,
                 lr_scheduler=None,
                 lr_scheduler_hparam: [dict, None] = None,
                 logger: StepLogger = None):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param critic: advantage estimation function $A(s,a) = Q(s,a) - V(s)$
        :param max_iter: number of iterations (policy updates)
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param num_epoch: number of iterations over all gathered samples during one policy update
        :param eps_clip: max/min probability ratio, see [1]
        :param value_fcn_coeff: weighting factor of the value function term in the combined loss, specific to PPO2
        :param entropy_coeff: weighting factor of the entropy term in the combined loss, specific to PPO2
        :param batch_size: number of samples per policy update batch
        :param std_init: initial standard deviation on the actions for the exploration noise
        :param num_workers: number of environments for parallel sampling
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created

        .. note::
            The Adam optimizer computes individual learning rates for all parameters. Thus, the learning rate scheduler
            schedules the maximum learning rate.
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)

        # Call ActorCritic's constructor
        super().__init__(env, policy, critic, save_dir, max_iter, logger)
        critic.standardize_adv = True  # enforce this for PPO2

        # Store the inputs
        self.num_epoch = num_epoch
        self.eps_clip = eps_clip
        self.value_fcn_coeff = value_fcn_coeff
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Initialize
        self.log_loss = True
        self._expl_strat = NormalActNoiseExplStrat(self._policy, std_init=std_init)
        self.sampler = ParallelRolloutSampler(
            env, self._expl_strat,
            num_workers=num_workers,
            min_steps=min_steps,
            min_rollouts=min_rollouts
        )
        self.optim = to.optim.Adam(
            [{'params': self._expl_strat.policy.parameters()},
             {'params': self._expl_strat.noise.parameters()},
             {'params': self._critic.value_fcn.parameters()}],
            lr=lr, eps=1e-5
        )
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self._expl_strat

    def loss_fcn(self, log_probs: to.Tensor, log_probs_old: to.Tensor, adv: to.Tensor,
                 v_pred: to.Tensor, v_pred_old: to.Tensor, v_targ: to.Tensor) -> to.Tensor:
        """
        PPO2 loss function

        :param log_probs: logarithm of the probabilities of the taken actions using the updated policy
        :param log_probs_old: logarithm of the probabilities of the taken actions using the old policy
        :param adv: advantage values
        :param v_pred: predicted value function values
        :param v_pred_old: predicted value function values using the old value function
        :param v_targ: target value function values
        :return: combined loss value
        """
        v_pred = v_pred.to(self.policy.device)
        v_targ = v_targ.to(self.policy.device)
        v_pred_old = v_pred_old.to(self.policy.device)
        # Policy loss
        prob_ratio = to.exp(log_probs - log_probs_old).to(self.policy.device)
        pr_clip = prob_ratio.clamp(1 - self.eps_clip, 1 + self.eps_clip)
        adv = adv.to(self.policy.device)
        p_loss1 = -adv*prob_ratio
        p_loss2 = -adv*pr_clip
        policy_loss = to.mean(to.max(p_loss1, p_loss2))

        # Value function loss
        v_pred_diffs = v_pred.to(self.policy.device) - v_pred_old.to(self.policy.device)
        v_pred_clip = v_pred_old + v_pred_diffs.clamp(-self.eps_clip, self.eps_clip)
        v_loss1 = to.pow(v_targ - v_pred, 2)
        v_loss2 = to.pow(v_targ - v_pred_clip, 2)
        value_fcn_loss = 0.5*to.mean(to.max(v_loss1, v_loss2))

        # Current entropy of the exploration strategy (was constant over the rollout)
        entropy = self._expl_strat.noise.get_entropy()

        # Return the combined loss
        return policy_loss + self.value_fcn_coeff*value_fcn_loss - self.entropy_coeff*entropy

    def update(self, rollouts: Sequence[StepSequence]):
        # Turn the batch of rollouts into a list of steps
        concat_ros = StepSequence.concat(rollouts)
        concat_ros.torch(data_type=to.get_default_dtype())

        with to.no_grad():
            # Compute the action probabilities using the old (before update) policy
            act_stats = compute_action_statistics(concat_ros, self._expl_strat)
            log_probs_old = act_stats.log_probs
            act_distr_old = act_stats.act_distr

            # Compute value predictions using the old old (before update) value function
            v_pred_old = self._critic.values(concat_ros)

        # Attach advantages and old log probs to rollout
        concat_ros.add_data('log_probs_old', log_probs_old)
        concat_ros.add_data('v_pred_old', v_pred_old)

        # For logging the gradient norms
        policy_grad_norm = []
        value_fcn_grad_norm = []

        # Compute the value targets (empirical discounted returns) for all samples before fitting the V-fcn parameters
        adv = self._critic.gae(concat_ros)  # done with to.no_grad()
        v_targ = discounted_values(rollouts, self._critic.gamma).view(-1, 1)  # empirical discounted returns
        concat_ros.add_data('adv', adv)
        concat_ros.add_data('v_targ', v_targ)

        # Iterations over the whole data set
        for e in range(self.num_epoch):

            for batch in tqdm(concat_ros.split_shuffled_batches(
                self.batch_size,
                complete_rollouts=self._policy.is_recurrent or isinstance(self._critic.value_fcn, RecurrentPolicy)),
                total=num_iter_from_rollouts(None, concat_ros, self.batch_size),
                desc=f'Epoch {e}', unit='batches', file=sys.stdout, leave=False):
                # Reset the gradients
                self.optim.zero_grad()

                # Compute log of the action probabilities for the mini-batch
                log_probs = compute_action_statistics(batch, self._expl_strat).log_probs.to(self.policy.device)

                # Compute value predictions for the mini-batch
                v_pred = self._critic.values(batch)

                # Compute combined loss and backpropagate
                loss = self.loss_fcn(log_probs, batch.log_probs_old, batch.adv, v_pred, batch.v_pred_old, batch.v_targ)
                loss.backward()

                # Clip the gradients if desired
                policy_grad_norm.append(self.clip_grad(self._expl_strat.policy, self.max_grad_norm))
                value_fcn_grad_norm.append(self.clip_grad(self._critic.value_fcn, self.max_grad_norm))

                # Call optimizer
                self.optim.step()

                if to.isnan(self._expl_strat.noise.std).any():
                    raise RuntimeError(f'At least one exploration parameter became NaN! The exploration parameters are'
                                       f'\n{self._expl_strat.std.detach().cpu().numpy()}')

            # Update the learning rate if a scheduler has been specified
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

        # Additional logging
        if self.log_loss:
            with to.no_grad():
                # Compute value predictions using the new (after the updates) value function approximator
                v_pred = self._critic.values(concat_ros).to(self.policy.device)
                v_loss_old = self._critic.loss_fcn(v_pred_old.to(self.policy.device), v_targ.to(self.policy.device)).to(
                    self.policy.device)
                v_loss_new = self._critic.loss_fcn(v_pred, v_targ).to(self.policy.device)
                value_fcn_loss_impr = v_loss_old - v_loss_new  # positive values are desired

                # Compute the action probabilities using the new (after the updates) policy
                act_stats = compute_action_statistics(concat_ros, self._expl_strat)
                log_probs_new = act_stats.log_probs
                act_distr_new = act_stats.act_distr
                loss_after = self.loss_fcn(log_probs_new, log_probs_old, adv, v_pred, v_pred_old, v_targ)
                kl_avg = to.mean(kl_divergence(act_distr_old, act_distr_new))  # mean seeking a.k.a. inclusive KL

                # Compute explained variance (after the updates)
                explvar = explained_var(v_pred, v_targ)
                self.logger.add_value('explained var', explvar.detach().cpu().numpy())
                self.logger.add_value('V-fcn loss improvement', value_fcn_loss_impr.detach().cpu().numpy())
                self.logger.add_value('loss after', loss_after.detach().cpu().numpy())
                self.logger.add_value('KL(old_new)', kl_avg.item())

        # Logging
        self.logger.add_value('avg expl strat std', to.mean(self._expl_strat.noise.std.data).detach().cpu().numpy())
        self.logger.add_value('expl strat entropy', self._expl_strat.noise.get_entropy().item())
        self.logger.add_value('avg policy grad norm', np.mean(policy_grad_norm))
        self.logger.add_value('avg V-fcn grad norm', np.mean(value_fcn_grad_norm))
        if self._lr_scheduler is not None:
            self.logger.add_value('learning rate', self._lr_scheduler.get_lr())
