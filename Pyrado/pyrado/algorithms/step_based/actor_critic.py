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
from abc import ABC, abstractmethod
from typing import Sequence

import pyrado
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.base import Algorithm
from pyrado.environments.base import Env
from pyrado.logger.step import StepLogger
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.policies.base import Policy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.input_output import print_cbt


class ActorCritic(Algorithm, ABC):
    """ Base class of all actor critic algorithms """

    def __init__(
        self, env: Env, actor: Policy, critic: GAE, save_dir: pyrado.PathLike, max_iter: int, logger: StepLogger = None
    ):
        """
        Constructor

        :param env: the environment which the policy operates
        :param actor: policy taking the actions in the environment
        :param critic: estimates the value of states (e.g. advantage or return)
        :param save_dir: directory to save the snapshots i.e. the results in
        :param max_iter: maximum number of iterations
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
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

        # Log metrics computed from the old policy (before the update)
        all_lengths = np.array([ro.length for ro in ros])
        self._cnt_samples += int(np.sum(all_lengths))
        rets = [ro.undiscounted_return() for ro in ros]
        self.logger.add_value("max return", np.max(rets), 4)
        self.logger.add_value("median return", np.median(rets), 4)
        self.logger.add_value("avg return", np.mean(rets), 4)
        self.logger.add_value("min return", np.min(rets), 4)
        self.logger.add_value("std return", np.std(rets), 4)
        self.logger.add_value("avg rollout len", np.mean(all_lengths), 4)
        self.logger.add_value("num total samples", self._cnt_samples)

        # Save snapshot data
        self.make_snapshot(snapshot_mode, float(np.mean(rets)), meta_info)

        # Update the advantage estimator and the policy
        self.update(ros)

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

    def init_modules(self, warmstart: bool, suffix: str = "", prefix: str = None, **kwargs):
        if prefix is None:
            prefix = f"iter_{self._curr_iter - 1}"

        ppi = kwargs.get("policy_param_init", None)
        vpi = kwargs.get("valuefcn_param_init", None)

        if warmstart and ppi is not None and vpi is not None:
            self._policy.init_param(ppi)
            self._critic.vfcn.init_param(vpi)
            print_cbt("Learning given an fixed parameter initialization.", "w")

        elif warmstart and ppi is None and self._curr_iter > 0:
            self._policy = pyrado.load("policy.pt", self.save_dir, prefix=prefix, suffix=suffix, obj=self._policy)
            self._critic.vfcn = pyrado.load(
                "vfcn.pt", self.save_dir, prefix=prefix, suffix=suffix, obj=self._critic.vfcn
            )
            print_cbt(f"Learning given the results from iteration {self._curr_iter - 1}", "w")

        else:
            # Reset the policy
            self._policy.init_param()
            self._critic.vfcn.init_param()
            print_cbt("Learning from scratch.", "w")

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            pyrado.save(self._env, "env.pkl", self.save_dir)
            pyrado.save(self._expl_strat.policy, "policy.pt", self.save_dir, use_state_dict=True)
            pyrado.save(self._critic.vfcn, "vfcn.pt", self.save_dir, use_state_dict=True)

        else:
            # This algorithm instance is a subroutine of another algorithm
            prefix = meta_info.get("prefix", "")
            suffix = meta_info.get("suffix", "")
            pyrado.save(
                self._expl_strat.policy, "policy.pt", self.save_dir, prefix=prefix, suffix=suffix, use_state_dict=True
            )
            pyrado.save(self._critic.vfcn, "vfcn.pt", self.save_dir, prefix=prefix, suffix=suffix, use_state_dict=True)
