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
from copy import deepcopy
from typing import Optional, Tuple

import numpy as np
import torch as to
import torch.nn as nn
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.value_based import ValueBased
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import EpsGreedyExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.policies.feed_back.fnn import DiscreteActQValPolicy
from pyrado.sampling.cvar_sampler import CVaRSampler
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler


class DQL(ValueBased):
    """
    Deep Q-Learning (without bells and whistles)

    .. seealso::
        [1] V. Mnih et.al., "Human-level control through deep reinforcement learning", Nature, 2015
    """

    name: str = "dql"

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env: Env,
        policy: DiscreteActQValPolicy,
        memory_size: int,
        eps_init: float,
        eps_schedule_gamma: float,
        gamma: float,
        max_iter: int,
        num_updates_per_step: int,
        target_update_intvl: Optional[int] = 5,
        num_init_memory_steps: Optional[int] = None,
        min_rollouts: Optional[int] = None,
        min_steps: Optional[int] = None,
        batch_size: int = 256,
        eval_intvl: int = 100,
        max_grad_norm: float = 0.5,
        lr: float = 5e-4,
        lr_scheduler=None,
        lr_scheduler_hparam: Optional[dict] = None,
        num_workers: int = 4,
        logger: Optional[StepLogger] = None,
        use_trained_policy_for_refill: bool = False,
    ):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: (current) Q-network updated by this algorithm
        :param memory_size: number of transitions in the replay memory buffer
        :param eps_init: initial value for the probability of taking a random action, constant if `eps_schedule_gamma=1`
        :param eps_schedule_gamma: temporal discount factor for the exponential decay of epsilon
        :param gamma: temporal discount factor for the state values
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_updates_per_step: number of (batched) updates per algorithm steps
        :param target_update_intvl: number of iterations that pass before updating the `qfcn_targ` network
        :param num_init_memory_steps: number of samples used to initially fill the replay buffer with, pass `None` to
                                      fill the buffer completely
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param batch_size: number of samples per policy update batch
        :param eval_intvl: interval in which the evaluation rollouts are collected, also the interval in which the
                           logger prints the summary statistics
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param lr: (initial) learning rate for the optimizer which can be by modified by the scheduler.
                   By default, the learning rate is constant.
        :param lr_scheduler: learning rate scheduler that does one step per epoch (pass through the whole data set)
        :param lr_scheduler_hparam: hyper-parameters for the learning rate scheduler
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        :param use_trained_policy_for_refill: whether to use the trained policy instead of a dummy policy to refill the
                                              replay buffer after resets
        """
        if not isinstance(policy, DiscreteActQValPolicy):
            raise pyrado.TypeErr(given=policy, expected_type=DiscreteActQValPolicy)

        # Call ValueBased's constructor
        super().__init__(
            save_dir=save_dir,
            env=env,
            policy=policy,
            memory_size=memory_size,
            gamma=gamma,
            max_iter=max_iter,
            num_updates_per_step=num_updates_per_step,
            target_update_intvl=target_update_intvl,
            num_init_memory_steps=num_init_memory_steps,
            min_rollouts=min_rollouts,
            min_steps=min_steps,
            batch_size=batch_size,
            eval_intvl=eval_intvl,
            max_grad_norm=max_grad_norm,
            num_workers=num_workers,
            logger=logger,
            use_trained_policy_for_refill=use_trained_policy_for_refill,
        )

        self.qfcn_targ = deepcopy(self._policy).eval()  # will not be trained using the optimizer
        self.eps = eps_init

        # Create sampler for exploration during training
        self._expl_strat = EpsGreedyExplStrat(self._policy, eps_init, eps_schedule_gamma)
        self._sampler = ParallelRolloutSampler(
            self._env,
            self._expl_strat,
            num_workers=num_workers if min_steps != 1 else 1,
            min_steps=min_steps,
            min_rollouts=min_rollouts,
        )

        # Q-function optimizer
        self.optim = to.optim.RMSprop([{"params": self._policy.parameters()}], lr=lr)

        # Learning rate scheduler
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_hparam = lr_scheduler_hparam
        if lr_scheduler is not None:
            self._lr_scheduler = lr_scheduler(self.optim, **lr_scheduler_hparam)

    @property
    def sampler(self) -> ParallelRolloutSampler:
        return self._sampler

    @sampler.setter
    def sampler(self, sampler: ParallelRolloutSampler):
        if not isinstance(sampler, (ParallelRolloutSampler, CVaRSampler)):
            raise pyrado.TypeErr(given=sampler, expected_type=(ParallelRolloutSampler, CVaRSampler))
        self._sampler = sampler

    @staticmethod
    def loss_fcn(q_vals: to.Tensor, expected_q_vals: to.Tensor) -> to.Tensor:
        r"""
        The Huber loss function on the one-step TD error $\delta = Q(s,a) - (r + \gamma \max_a Q(s^\prime, a))$.

        :param q_vals: state-action values $Q(s,a)$, from policy network
        :param expected_q_vals: expected state-action values $r + \gamma \max_a Q(s^\prime, a)$, from target network
        :return: loss value
        """
        return nn.functional.smooth_l1_loss(q_vals, expected_q_vals)

    def update(self):
        """Update the policy's and qfcn_targ Q-function's parameters on transitions sampled from the replay memory."""
        losses = to.zeros(self.num_batch_updates)
        policy_grad_norm = to.zeros(self.num_batch_updates)

        for b in tqdm(
            range(self.num_batch_updates),
            total=self.num_batch_updates,
            desc=f"Updating",
            unit="batches",
            file=sys.stdout,
            leave=False,
        ):

            # Sample steps and the associated next step from the replay memory
            steps, next_steps = self._memory.sample(self.batch_size)
            steps.torch(data_type=to.get_default_dtype())
            next_steps.torch(data_type=to.get_default_dtype())

            # Create masks for the non-final observations
            not_done = to.from_numpy(1.0 - steps.done).to(device=self.policy.device, dtype=to.get_default_dtype())

            # Compute the state-action values Q(s,a) using the current DQN policy
            q_vals = self.expl_strat.policy.q_values_argmax(steps.observations)

            # Compute the second term of TD-error
            with to.no_grad():
                next_v_vals = self.qfcn_targ.q_values_argmax(next_steps.observations)
                expected_q_val = steps.rewards.to(self.policy.device) + not_done * self.gamma * next_v_vals

            # Compute the loss, clip the gradients if desired, and do one optimization step
            loss = DQL.loss_fcn(q_vals, expected_q_val)
            losses[b] = loss.data
            self.optim.zero_grad()
            loss.backward()
            policy_grad_norm[b] = Algorithm.clip_grad(self.expl_strat.policy, self.max_grad_norm)
            self.optim.step()

            # Update the qfcn_targ network by copying all weights and biases from the DQN policy
            if (self._curr_iter * self.num_batch_updates + b) % self.target_update_intvl == 0:
                self.qfcn_targ.load_state_dict(self.expl_strat.policy.state_dict())

        # Schedule the exploration parameter epsilon
        self.expl_strat.schedule_eps(self._curr_iter)

        # Update the learning rate if a scheduler has been specified
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()

        # Logging
        with to.no_grad():
            self.logger.add_value("loss after", to.mean(losses), 4)
        self.logger.add_value("expl strat eps", self.expl_strat.eps, 4)
        self.logger.add_value("avg grad norm policy", to.mean(policy_grad_norm), 4)
        if self._lr_scheduler is not None:
            self.logger.add_value("avg lr", np.mean(self._lr_scheduler.get_last_lr()), 6)

    def reset(self, seed: Optional[int] = None):
        # Reset samplers, replay memory, exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Reset the learning rate scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.last_epoch = -1

    def init_modules(self, warmstart: bool, suffix: str = "", prefix: str = "", **kwargs):
        # Initialize the policy
        super().init_modules(warmstart, suffix, prefix, **kwargs)

        if prefix == "":
            prefix = f"iter_{self._curr_iter - 1}"

        tpi = kwargs.get("target_param_init", None)

        if warmstart and tpi is not None:
            self.qfcn_targ.init_param(tpi)
        elif warmstart and tpi is None and self._curr_iter > 0:
            self.qfcn_targ = pyrado.load(
                "qfcn_target.pt", self.save_dir, prefix=prefix, suffix=suffix, obj=self.qfcn_targ
            )
        else:
            # Reset the target Q-function
            self.qfcn_targ.init_param()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self.qfcn_targ, "qfcn_target.pt", self.save_dir, use_state_dict=True)

        else:
            # This algorithm instance is a subroutine of another algorithm
            pyrado.save(
                self.qfcn_targ,
                "qfcn_target.pt",
                self.save_dir,
                prefix=meta_info.get("prefix", ""),
                suffix=meta_info.get("suffix", ""),
                use_state_dict=True,
            )

    def load_snapshot(self, parsed_args) -> Tuple[Env, Policy, dict]:
        env, policy, extra = super().load_snapshot(parsed_args)

        # Algorithm specific
        ex_dir = self._save_dir or getattr(parsed_args, "dir", None)
        if self.name == "dql":
            extra["qfcn_target"] = pyrado.load("qfcn_target.pt", ex_dir, obj=self.qfcn_targ, verbose=True)

        return env, policy, extra
