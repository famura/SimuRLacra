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

import sys
import numpy as np
import torch as to
from torch.distributions.kl import kl_divergence
from tqdm import tqdm
from typing import Sequence

from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.utils import compute_action_statistics, num_iter_from_rollouts
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.recurrent.base import RecurrentPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.step_sequence import StepSequence, discounted_values
from pyrado.utils.math import explained_var


class A2C(ActorCritic):
    """ Advantage Actor Critic (A2C) """

    name: str = 'a2c'

    def __init__(self,
                 save_dir: str,
                 env: Env,
                 policy: Policy,
                 critic: GAE,
                 max_iter: int,
                 min_rollouts: int = None,
                 min_steps: int = None,
                 value_fcn_coeff: float = 0.5,
                 entropy_coeff: float = 1e-3,
                 batch_size: int = 32,
                 std_init: float = 1.0,
                 max_grad_norm: float = None,
                 num_workers: int = 4,
                 lr: float = 5e-4,
                 lr_scheduler=None,
                 lr_scheduler_hparam: [dict, None] = None,
                 logger: StepLogger = None):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param critic: advantage estimation function $A(s,a) = Q(s,a) - V(s)$
        :param max_iter: number of iterations (policy updates)
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param value_fcn_coeff: weighting factor of the value function term in the combined loss, specific to PPO2
        :param entropy_coeff: weighting factor of the entropy term in the combined loss, specific to PPO2
        :param batch_size: number of samples per policy update batch
        :param std_init: initial standard deviation on the actions for the exploration noise
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param num_workers: number of environments for parallel sampling
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        # Call ActorCritic's constructor
        super().__init__(env, policy, critic, save_dir, max_iter, logger)

        # Store the inputs
        self.min_rollouts = min_rollouts
        self.min_steps = min_steps
        self.value_fcn_coeff = value_fcn_coeff
        self.entropy_coeff = entropy_coeff
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        # Initialize
        self._expl_strat = NormalActNoiseExplStrat(self._policy, std_init=std_init)
        self.sampler = ParallelRolloutSampler(
            env, self.expl_strat,
            num_workers=num_workers,
            min_steps=min_steps,
            min_rollouts=min_rollouts
        )
        self.optim = to.optim.RMSprop(
            [{'params': self._policy.parameters()},
             {'params': self.expl_strat.noise.parameters()},
             {'params': self._critic.value_fcn.parameters()}],
            lr=lr, eps=1e-5
        )
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    def loss_fcn(self, log_probs: to.Tensor, adv: to.Tensor, v_pred: to.Tensor, v_targ: to.Tensor):
        """
        A2C loss function

        :param log_probs: logarithm of the probabilities of the taken actions
        :param adv: advantage values
        :param v_pred: predicted value function values
        :param v_targ: target value function values
        :return: combined loss value
        """
        # Policy, value function, and entropy losses
        policy_loss = -to.mean(adv.to(self.policy.device)*log_probs)
        value_fcn_loss = 0.5*to.mean(to.pow(v_targ.to(self.policy.device) - v_pred.to(self.policy.device), 2))  # former v_targ.cpu() - v_pred.cpu()
        entropy_mean = to.mean(self.expl_strat.noise.get_entropy())

        # Return the combined loss
        return policy_loss + self.value_fcn_coeff*value_fcn_loss - self.entropy_coeff*entropy_mean

    def update(self, rollouts: Sequence[StepSequence]):
        # Turn the batch of rollouts into a list of steps
        concat_ros = StepSequence.concat(rollouts)
        concat_ros.torch(data_type=to.get_default_dtype())

        # Compute the value targets (empirical discounted returns) for all samples before fitting the V-fcn parameters
        adv = self._critic.gae(concat_ros)  # done with to.no_grad()
        v_targ = discounted_values(rollouts, self._critic.gamma).view(-1, 1).to(self.policy.device)  # empirical discounted returns

        with to.no_grad():
            # Compute value predictions and the GAE using the old (before the updates) value function approximator
            v_pred = self._critic.values(concat_ros)

            # Compute the action probabilities using the old (before update) policy
            act_stats = compute_action_statistics(concat_ros, self._expl_strat)
            log_probs_old = act_stats.log_probs
            act_distr_old = act_stats.act_distr
            loss_before = self.loss_fcn(log_probs_old, adv, v_pred, v_targ)
            self.logger.add_value('loss before', loss_before, 4)

        concat_ros.add_data('adv', adv)
        concat_ros.add_data('v_targ', v_targ)

        # For logging the gradients' norms
        policy_grad_norm = []

        for batch in tqdm(concat_ros.split_shuffled_batches(
            self.batch_size,
            complete_rollouts=self._policy.is_recurrent or isinstance(self._critic.value_fcn, RecurrentPolicy)),
            total=num_iter_from_rollouts(None, concat_ros, self.batch_size),
            desc='Updating', unit='batches', file=sys.stdout, leave=False):
            # Reset the gradients
            self.optim.zero_grad()

            # Compute log of the action probabilities for the mini-batch
            log_probs = compute_action_statistics(batch, self._expl_strat).log_probs

            # Compute value predictions for the mini-batch
            v_pred = self._critic.values(batch)

            # Compute combined loss and backpropagate
            loss = self.loss_fcn(log_probs, batch.adv, v_pred, batch.v_targ)
            loss.backward()

            # Clip the gradients if desired
            policy_grad_norm.append(self.clip_grad(self.expl_strat.policy, self.max_grad_norm))

            # Call optimizer
            self.optim.step()

        # Update the learning rate if a scheduler has been specified
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        if to.isnan(self.expl_strat.noise.std).any():
            raise RuntimeError(f'At least one exploration parameter became NaN! The exploration parameters are'
                               f'\n{self.expl_strat.std.item()}')

        # Logging
        with to.no_grad():
            # Compute value predictions and the GAE using the new (after the updates) value function approximator
            v_pred = self._critic.values(concat_ros).to(self.policy.device)
            adv = self._critic.gae(concat_ros)  # done with to.no_grad()

            # Compute the action probabilities using the new (after the updates) policy
            act_stats = compute_action_statistics(concat_ros, self._expl_strat)
            log_probs_new = act_stats.log_probs
            act_distr_new = act_stats.act_distr
            loss_after = self.loss_fcn(log_probs_new, adv, v_pred, v_targ)
            kl_avg = to.mean(
                kl_divergence(act_distr_old, act_distr_new))  # mean seeking a.k.a. inclusive KL
            explvar = explained_var(v_pred, v_targ)  # values close to 1 are desired
            self.logger.add_value('loss after', loss_after, 4)
            self.logger.add_value('KL(old_new)', kl_avg, 4)
            self.logger.add_value('explained var', explvar, 4)

        ent = self.expl_strat.noise.get_entropy()
        self.logger.add_value('avg expl strat std', to.mean(self.expl_strat.noise.std), 4)
        self.logger.add_value('expl strat entropy', to.mean(ent), 4)
        self.logger.add_value('avg policy grad norm', np.mean(policy_grad_norm), 4)
        if self._lr_scheduler is not None:
            self.logger.add_value('avg learning rate', np.mean(self._lr_scheduler.get_lr()), 6)
