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

from typing import Union

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environment_wrappers.adversarial import (
    AdversarialDynamicsWrapper,
    AdversarialObservationWrapper,
    AdversarialStateWrapper,
)
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.environments.sim_base import SimEnv
from pyrado.exploration.stochastic_action import StochasticActionExplStrat
from pyrado.logger.step import StepLogger
from pyrado.policies.base import Policy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.sequences import *


class ARPL(Algorithm):
    """
    Adversarially Robust Policy Learning (ARPL)

    .. seealso::
        A. Mandlekar, Y. Zhu, A. Garg, L. Fei-Fei, S. Savarese, "Adversarially Robust Policy Learning:
        Active Construction of Physically-Plausible Perturbations", IROS, 2017
    """

    name: str = "arpl"

    def __init__(
        self,
        save_dir: pyrado.PathLike,
        env: Union[SimEnv, StateAugmentationWrapper],
        subrtn: Algorithm,
        policy: Policy,
        expl_strat: StochasticActionExplStrat,
        max_iter: int,
        num_rollouts: int = None,
        steps_num: int = None,
        apply_dynamics_noise: bool = False,
        logger: StepLogger = None,
    ):
        """
        Constructor

        :param save_dir: directory to save the snapshots i.e. the results in
        :param env: the environment in which the agent should be trained
        :param subrtn: algorithm which performs the policy / value-function optimization
        :param policy: policy to be updated
        :param expl_strat: the exploration strategy
        :param max_iter: the maximum number of iterations
        :param num_rollouts: the number of rollouts to be performed for each update step
        :param steps_num: the number of steps to be performed for each update step
        :param apply_dynamics_noise: whether adversarially generated dynamics noise should be applied
        :param dyn_eps: the intensity of generated dynamics noise
        :param dyn_phi: the probability of applying dynamics noise
        :param halfspan: the halfspan of the uniform random distribution used to sample
        :param apply_proccess_noise: whether adversarially generated process noise should be applied
        :param proc_eps: the intensity of generated process noise
        :param proc_phi: the probability of applying process noise
        :param apply_observation_noise: whether adversarially generated observation noise should be applied
        :param obs_eps: the intensity of generated observation noise
        :param obs_phi: the probability of applying observation noise
        :param logger: logger for every step of the algorithm, if `None` the default logger will be created
        """
        assert isinstance(subrtn, Algorithm)
        assert isinstance(max_iter, int) and max_iter > 0

        super().__init__(save_dir, max_iter, policy, logger)

        self._env = env

        # Subroutine
        self._subrtn = subrtn
        self._subrtn.save_name = "subrtn"

        pyrado.save(self._env, "env.pkl", self.save_dir)

    @staticmethod
    def wrap_env(
        env,
        policy,
        dynamics=False,
        process=False,
        observation=False,
        dyn_eps: float = 0.01,
        dyn_phi: float = 0.1,
        halfspan: float = 0.25,
        proc_eps: float = 0.01,
        proc_phi: float = 0.05,
        obs_eps: float = 0.01,
        obs_phi: float = 0.05,
    ):
        # Initialize adversarial wrappers
        if dynamics:
            assert isinstance(env, StateAugmentationWrapper)
            env = AdversarialDynamicsWrapper(env, policy, dyn_eps, dyn_phi, halfspan)
        if process:
            env = AdversarialStateWrapper(env, policy, proc_eps, proc_phi)
        if observation:
            env = AdversarialObservationWrapper(env, policy, obs_eps, obs_phi)
        return env

    @property
    def sample_count(self) -> int:
        return self._subrtn.sample_count

    def step(self, snapshot_mode: str, meta_info: dict = None):
        rollouts = self._subrtn.sampler.sample()
        rets = [ro.undiscounted_return() for ro in rollouts]
        ret_avg = np.mean(rets)
        ret_med = np.median(rets)
        ret_std = np.std(rets)
        self.logger.add_value("avg return", ret_avg)
        self.logger.add_value("median return", ret_med)
        self.logger.add_value("std return", ret_std)
        self.logger.add_value("num total samples", self._cnt_samples)
        self.logger.add_value("avg rollout len", np.mean([ro.length for ro in rollouts]))

        # Sub-routine
        self._subrtn.update(rollouts)
        self._subrtn.logger.record_step()
        self._subrtn.make_snapshot(snapshot_mode, ret_avg.item())

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of a meta-algorithm
            self._subrtn.save_snapshot(meta_info)
        else:
            raise pyrado.ValueErr(msg=f"{self.name} is not supposed be run as a subroutine!")
