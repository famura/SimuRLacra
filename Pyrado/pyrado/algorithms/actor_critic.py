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
import torch as to
from abc import ABC, abstractmethod
from typing import Sequence

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.utils import save_prefix_suffix, load_prefix_suffix
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.policies.base import Policy
from pyrado.sampling.step_sequence import StepSequence


class ActorCritic(Algorithm, ABC):
    """ Base class of all actor critic algorithms """

    def __init__(self,
                 env: Env,
                 actor: Policy,
                 critic: GAE,
                 save_dir: str,
                 max_iter: int,
                 logger: StepLogger = None):
        """
        Constructor

        :param env: the environment which the policy operates
        :param actor: policy taking the actions in the environment
        :param critic: estimates the value of states (e.g. advantage or return)
        :param save_dir: directory to save the snapshots i.e. the results in
        :param max_iter: maximum number of iterations
        :param logger: logger for every step of the algorithm
        """
        if not isinstance(env, Env):
            raise pyrado.TypeErr(given=env, expected_type=Env)
        if not isinstance(critic, GAE):
            raise pyrado.TypeErr(given=critic, expected_type=GAE)

        # Call Algorithm's constructor
        super().__init__(save_dir, max_iter, actor, logger)

        # Store the inputs
        self._env = env
        self._critic = critic

        # Initialize
        self._expl_strat = None
        self.sampler = None
        self._lr_scheduler = None
        self._lr_scheduler_hparam = None

    @property
    def critic(self) -> GAE:
        """ Get the critic. """
        return self._critic

    @critic.setter
    def critic(self, critic: GAE):
        """ Set the critic. """
        if not isinstance(critic, GAE):
            pyrado.TypeErr(given=critic, expected_type=GAE)
        self._critic = critic

    @property
    def expl_strat(self) -> NormalActNoiseExplStrat:
        return self._expl_strat

    def step(self, snapshot_mode: str, meta_info: dict = None):
        # Sample rollouts
        ros = self.sampler.sample()

        # Log return-based metrics
        rets = [ro.undiscounted_return() for ro in ros]
        ret_min = np.min(rets)
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_max = np.max(rets)
        ret_std = np.std(rets)
        self.logger.add_value('max return', ret_max)
        self.logger.add_value('median return', ret_med)
        self.logger.add_value('avg return', ret_avg)
        self.logger.add_value('min return', ret_min)
        self.logger.add_value('std return', ret_std)
        self.logger.add_value('num rollouts', len(ros))
        self.logger.add_value('avg rollout len', np.mean([ro.length for ro in ros]))

        # Update the advantage estimator and the policy
        self.update(ros)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(ret_avg), meta_info)

    @abstractmethod
    def update(self, rollouts: Sequence[StepSequence]):
        """
        Update the actor and critic parameters from the given batch of rollouts.

        :param rollouts: batch of rollouts
        """
        raise NotImplementedError

    def reset(self, seed: int = None):
        # Reset the exploration strategy, internal variables and the random seeds
        super().reset(seed)

        # Re-initialize sampler in case env or policy changed
        self.sampler.reinit()

        # Reset the critic (also resets its learning rate scheduler)
        self.critic.reset()

        # Reset the learning rate scheduler
        if self._lr_scheduler is not None:
            self._lr_scheduler.last_epoch = -1

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        save_prefix_suffix(self._expl_strat.policy, 'policy', 'pt', self._save_dir, meta_info)
        save_prefix_suffix(self._critic.value_fcn, 'valuefcn', 'pt', self._save_dir, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            joblib.dump(self._env, osp.join(self._save_dir, 'env.pkl'))

    def load_snapshot(self, load_dir: str = None, meta_info: dict = None):
        # Get the directory to load from
        ld = load_dir if load_dir is not None else self._save_dir

        super().load_snapshot(ld, meta_info)
        self._critic.value_fcn = load_prefix_suffix(self._critic.value_fcn, 'valuefcn', 'pt', ld, meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._env = joblib.load(osp.join(ld, 'env.pkl'))
