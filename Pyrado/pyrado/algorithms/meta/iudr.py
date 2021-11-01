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
import torch as to

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.stopping_criteria.rollout_based_criteria import MinReturnStoppingCriterion
from pyrado.algorithms.utils import RolloutSavingWrapper
from pyrado.domain_randomization.domain_parameter import SelfPacedDomainParam
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env


class IUDR(Algorithm):
    """
    Incremental Uniform Domain Randomization (IUDR).

    This is an ablation of SPDR in the sense that the optimization is omitted and the contextual distribution is naively
    updated in fixed steps, disregarding the performance information.
    """

    name: str = "iudr"

    def __init__(
        self,
        env: DomainRandWrapper,
        subroutine: Algorithm,
        max_iter: int,
        performance_threshold: float,
        param_adjustment_portion: float = 0.9,
    ):
        """
        Constructor

        :param env: environment wrapped in a `DomainRandWrapper`
        :param subroutine: algorithm which performs the policy/value-function optimization; note that this algorithm
                           must be capable of learning a sufficient policy in its maximum number of iterations
        :param max_iter: iterations of the IUDR algorithm (not for the subroutine); changing the domain parameter
                         distribution is done by linear interpolation over this many iterations
        :param performance_threshold: lower bound for the performance that has to be reached until the domain parameter
                                      randomization is changed
        :param param_adjustment_portion: what portion of the IUDR iterations should be spent on adjusting the domain
                                         parameter distributions; defaults to `90%`
        """
        if not isinstance(subroutine, Algorithm):
            raise pyrado.TypeErr(given=subroutine, expected_type=Algorithm)
        if not hasattr(subroutine, "sampler"):
            raise AttributeError("The subroutine must have a sampler attribute!")
        if not typed_env(env, DomainRandWrapper):
            raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapper)

        # Call Algorithm's constructor with the subroutine's properties
        super().__init__(subroutine.save_dir, max_iter, subroutine.policy, subroutine.logger)

        self._subrtn = subroutine
        # Wrap the sampler with a rollout saving wrapper for the stopping criterion
        self._subrtn.sampler = RolloutSavingWrapper(self._subrtn.sampler)
        self._subrtn.save_name = self._subrtn.name
        self._subrtn.stopping_criterion = self._subrtn.stopping_criterion | MinReturnStoppingCriterion(
            return_threshold=performance_threshold
        )

        self._env = env
        self._performance_threshold = performance_threshold
        self._param_adjustment_scale = param_adjustment_portion * max_iter

        self._parameter = None
        for param in env.randomizer.domain_params:
            if isinstance(param, SelfPacedDomainParam):
                if self._parameter is None:
                    self._parameter = param
                else:
                    raise pyrado.ValueErr(msg="randomizer contains more than one spl param")

    @property
    def sample_count(self) -> int:
        # Forward to subroutine
        return self._subrtn.sample_count

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Perform a step of IUDR. This includes training the subroutine and updating the context distribution accordingly.
        For a description of the parameters see `pyrado.algorithms.base.Algorithm.step`.
        """
        self.save_snapshot()

        for param_a_idx, param_a_name in enumerate(self._parameter.name):
            for param_b_idx, param_b_name in enumerate(self._parameter.name):
                self.logger.add_value(
                    f"context cov for {param_a_name}--{param_b_name}",
                    self._parameter.context_cov[param_a_idx, param_b_idx].item(),
                )
                self.logger.add_value(
                    f"context cov_chol for {param_a_name}--{param_b_name}",
                    self._parameter.context_cov_chol[param_a_idx, param_b_idx].item(),
                )
                if param_a_name == param_b_name:
                    self.logger.add_value(
                        f"context mean for {param_a_name}", self._parameter.context_mean[param_a_idx].item()
                    )
                    break

        self._subrtn.reset()
        # Also reset the rollouts to not stop too early because the stopping criterion is fulfilled
        self._subrtn.sampler.reset_rollouts()
        self._subrtn.train(snapshot_mode, None, meta_info)

        # Prevents the parameters from overshooting the target
        if self.curr_iter >= self._param_adjustment_scale:
            context_mean_new = self._parameter.target_mean
            context_cov_chol_new = self._parameter.target_cov_chol
        else:
            context_mean_new = (
                self._parameter.context_mean
                + (self._parameter.target_mean - self._parameter.init_mean) / self._param_adjustment_scale
            )
            context_cov_chol_new = (
                self._parameter.context_cov_chol
                + (self._parameter.target_cov_chol - self._parameter.init_cov_chol) / self._param_adjustment_scale
            )
        self._parameter.adapt("context_mean", context_mean_new)
        self._parameter.adapt("context_cov_chol", context_cov_chol_new)

    def reset(self, seed: int = None):
        # Forward to subroutine
        self._subrtn.reset(seed)

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm
            self._subrtn.save_snapshot(meta_info)
