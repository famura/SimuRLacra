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

import os.path as osp
from abc import ABC, abstractmethod
from math import ceil
from typing import Optional, Union

import numpy as np

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import ReplayMemory
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import EpsGreedyExplStrat, SACExplStrat
from pyrado.logger.step import ConsolePrinter, CSVPrinter, StepLogger, TensorBoardPrinter
from pyrado.policies.base import Policy, TwoHeadedPolicy
from pyrado.policies.feed_forward.dummy import DummyPolicy, RecurrentDummyPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.utils.input_output import print_cbt_once


class ValueBased(Algorithm, ABC):
    """Base class of all value-based algorithms"""

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env: Env,
        policy: Union[Policy, TwoHeadedPolicy],
        memory_size: int,
        gamma: float,
        max_iter: int,
        num_updates_per_step: int,
        target_update_intvl: int,
        num_init_memory_steps: int,
        min_rollouts: int,
        min_steps: int,
        batch_size: int,
        eval_intvl: int,
        max_grad_norm: float,
        num_workers: int,
        logger: StepLogger,
    ):
        r"""
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment which the policy operates
        :param policy: policy to be updated
        :param memory_size: number of transitions in the replay memory buffer, e.g. 1000000
        :param gamma: temporal discount factor for the state values
        :param max_iter: maximum number of iterations (i.e. policy updates) that this algorithm runs
        :param num_updates_per_step: number of (batched) gradient updates per algorithm step
        :param target_update_intvl: number of iterations that pass before updating the target network
        :param num_init_memory_steps: number of samples used to initially fill the replay buffer with, pass `None` to
                                      fill the buffer completely
        :param min_rollouts: minimum number of rollouts sampled per policy update batch
        :param min_steps: minimum number of state transitions sampled per policy update batch
        :param batch_size: number of samples per policy update batch
        :param eval_intvl: interval in which the evaluation rollouts are collected, also the interval in which the
                           logger prints the summary statistics
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param num_workers: number of environments for parallel sampling
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(memory_size, int):
            raise pyrado.TypeErr(given=memory_size, expected_type=int)
        if not (num_init_memory_steps is None or isinstance(num_init_memory_steps, int)):
            raise pyrado.TypeErr(given=num_init_memory_steps, expected_type=int)

        if logger is None:
            # Create logger that only logs every logger_print_intvl steps of the algorithm
            logger = StepLogger(print_intvl=eval_intvl)
            logger.printers.append(ConsolePrinter())
            logger.printers.append(CSVPrinter(osp.join(save_dir, "progress.csv")))
            logger.printers.append(TensorBoardPrinter(osp.join(save_dir, "tb")))

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, policy, logger)

        self._env = env
        self._memory = ReplayMemory(memory_size)
        self.gamma = gamma
        self.target_update_intvl = target_update_intvl
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        if num_init_memory_steps is None:
            self.num_init_memory_steps = memory_size
        else:
            self.num_init_memory_steps = max(min(num_init_memory_steps, memory_size), batch_size)

        # Heuristic for number of gradient updates per step
        if num_updates_per_step is None:
            self.num_batch_updates = ceil(min_steps / env.max_steps) if min_steps is not None else min_rollouts
        else:
            self.num_batch_updates = num_updates_per_step

        # Create sampler for initial filling of the replay memory
        if policy.is_recurrent:
            self.init_expl_policy = RecurrentDummyPolicy(env.spec, policy.hidden_size)
        else:
            self.init_expl_policy = DummyPolicy(env.spec)
        self.sampler_init = ParallelRolloutSampler(
            self._env,
            self.init_expl_policy,
            num_workers=num_workers,
            min_steps=self.num_init_memory_steps,
        )

        # Create sampler for initial filling of the replay memory and evaluation
        self.sampler_eval = ParallelRolloutSampler(
            self._env,
            self._policy,
            num_workers=num_workers,
            min_steps=None,
            min_rollouts=100,
            show_progress_bar=True,
        )

        self._expl_strat = None  # must be implemented by subclass
        self._sampler = None  # must be implemented by subclass

        self._fill_with_init_sampler = True  # use the init sampler with the dummy policy on first run

    @property
    def expl_strat(self) -> Union[SACExplStrat, EpsGreedyExplStrat]:
        return self._expl_strat

    @property
    def memory(self) -> ReplayMemory:
        """Get the replay memory."""
        return self._memory

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if self._memory.isempty:
            # Warm-up phase
            print_cbt_once(f"Empty replay memory, collecting {self.num_init_memory_steps} samples.", "w")
            # Sample steps and store them in the replay memory
            if self._fill_with_init_sampler:
                ros = self.sampler_init.sample()
                self._fill_with_init_sampler = False
            else:
                # Save old bounds from the sampler
                min_rollouts = self.sampler.min_rollouts
                min_steps = self.sampler.min_steps
                # Set and sample with the init sampler settings
                self.sampler.set_min_count(min_steps=self.num_init_memory_steps)
                ros = self.sampler.sample()
                # Revert back to initial parameters
                self.sampler.set_min_count(min_rollouts=min_rollouts, min_steps=min_steps)
            self._memory.push(ros)
        else:
            # Sample steps and store them in the replay memory
            ros = self.sampler.sample()
            self._memory.push(ros)
        self._cnt_samples += sum([ro.length for ro in ros])  # don't count the evaluation samples

        # Log metrics computed from the old policy (before the update)
        if self._curr_iter % self.logger.print_intvl == 0:
            ros = self.sampler_eval.sample()
            rets = [ro.undiscounted_return() for ro in ros]
            ret_max = np.max(rets)
            ret_med = np.median(rets)
            ret_avg = np.mean(rets)
            ret_min = np.min(rets)
            ret_std = np.std(rets)
        else:
            ret_max, ret_med, ret_avg, ret_min, ret_std = 5 * [-pyrado.inf]  # dummy values
        self.logger.add_value("max return", ret_max, 4)
        self.logger.add_value("median return", ret_med, 4)
        self.logger.add_value("avg return", ret_avg, 4)
        self.logger.add_value("min return", ret_min, 4)
        self.logger.add_value("std return", ret_std, 4)
        self.logger.add_value("avg memory reward", self._memory.avg_reward(), 4)
        self.logger.add_value("avg rollout length", np.mean([ro.length for ro in ros]), 4)
        self.logger.add_value("num total samples", self._cnt_samples)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

        # Use data in the memory to update the policy and the Q-functions
        self.update()

    @abstractmethod
    def update(self):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None, fill_memory_with_dummy_policy: bool = False):
        """
        Reset the algorithm to its initial state. This should not reset learned policy parameters.
        By default, this resets the iteration count and the exploration strategy.
        Be sure to call this function if you override it.

        :param seed: seed value for the random number generators, pass `None` for no seeding
        :param fill_memory_with_dummy: if `True`, fill the memory with a random dummy policy instead of the trained policy
        """

        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize samplers in case env or policy changed
        self.sampler_init.reinit(self._env, self.init_expl_policy)
        self.sampler.reinit(self._env, self._expl_strat)
        self.sampler_eval.reinit(self._env, self._policy)

        # Optionally use the init sampler to fill memory buffer
        self._fill_with_init_sampler = fill_memory_with_dummy_policy

        # Reset the replay memory
        self._memory.reset()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self._env, "env.pkl", self.save_dir)
            pyrado.save(self._expl_strat.policy, "policy.pt", self.save_dir, use_state_dict=True)
        else:
            pyrado.save(
                self._expl_strat.policy,
                "policy.pt",
                self.save_dir,
                prefix=meta_info.get("prefix", ""),
                suffix=meta_info.get("suffix", ""),
                use_state_dict=True,
            )
