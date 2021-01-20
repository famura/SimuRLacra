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
import os.path as osp
from abc import ABC, abstractmethod
from math import ceil
from typing import Union, Optional

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import ReplayMemory
from pyrado.environments.base import Env
from pyrado.exploration.stochastic_action import SACExplStrat, EpsGreedyExplStrat
from pyrado.logger.step import StepLogger, ConsolePrinter, CSVPrinter, TensorBoardPrinter
from pyrado.policies.base import Policy, TwoHeadedPolicy
from pyrado.policies.special.dummy import RecurrentDummyPolicy, DummyPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.utils.input_output import print_cbt_once


class ValueBased(Algorithm, ABC):
    """ Base class of all value-based algorithms """

    def __init__(
        self,
        save_dir: str,
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
        num_workers: int,
        max_grad_norm: float,
        logger: StepLogger,
        logger_print_intvl: Optional[int] = 100,
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
        :param num_workers: number of environments for parallel sampling
        :param max_grad_norm: maximum L2 norm of the gradients for clipping, set to `None` to disable gradient clipping
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        :param logger_print_intvl: interval in which the logger prints
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(memory_size, int):
            raise pyrado.TypeErr(given=memory_size, expected_type=int)
        if not (num_init_memory_steps is None or isinstance(num_init_memory_steps, int)):
            raise pyrado.TypeErr(given=num_init_memory_steps, expected_type=int)

        if logger is None:
            # Create logger that only logs every logger_print_intvl steps of the algorithm
            logger = StepLogger(print_intvl=logger_print_intvl)
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
            self.num_init_memory_steps = min(num_init_memory_steps, memory_size)

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
            min_steps=10 * env.max_steps,
            min_rollouts=None,
            show_progress_bar=False,
        )

        self._expl_strat = None  # must be implemented by subclass
        self.sampler_trn = None  # must be implemented by subclass

    @property
    def expl_strat(self) -> Union[SACExplStrat, EpsGreedyExplStrat]:
        return self._expl_strat

    @property
    def memory(self) -> ReplayMemory:
        """ Get the replay memory. """
        return self._memory

    def step(self, snapshot_mode: str, meta_info: dict = None):
        if self._memory.isempty:
            # Warm-up phase
            print_cbt_once("Collecting samples until replay memory if full.", "w")
            # Sample steps and store them in the replay memory
            ros = self.sampler_init.sample()
            self._memory.push(ros)
        else:
            # Sample steps and store them in the replay memory
            ros = self.sampler_trn.sample()
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

        # Use data in the memory to update the policy and the Q-functions
        self.update()

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

    @abstractmethod
    def update(self):
        raise NotImplementedError

    def reset(self, seed: Optional[int] = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize samplers in case env or policy changed
        self.sampler_init.reinit(self._env, self.init_expl_policy)
        self.sampler_trn.reinit(self._env, self._expl_strat)
        self.sampler_eval.reinit(self._env, self._policy)

        # Reset the replay memory
        self._memory.reset()

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        pyrado.save(self._expl_strat.policy, "policy", "pt", self.save_dir, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self._env, "env", "pkl", self.save_dir, meta_info)
