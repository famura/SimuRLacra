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

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.stopping_criteria.rollout_based_criteria import MinReturnStoppingCriterion
from pyrado.algorithms.utils import RolloutSavingWrapper
from pyrado.domain_randomization.domain_parameter import SelfPacedDomainParam
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env


class IDR(Algorithm):
    """Iterative Domain Randomization (IDR)."""

    name: str = "idr"

    def __init__(
        self,
        env: DomainRandWrapper,
        subroutine: Algorithm,
        max_iter: int,
        performance_threshold: float,
        param_adjustment_portion: float = 0.9,
    ):
        """
        Constructor.

        :param env: environment wrapped in a DomainRandWrapper
        :param subroutine: algorithm which performs the policy/value-function optimization; note that this algorithm
                           must be capable of learning a sufficient policy in its maximum number of iterations
        :param max_iter: iterations of the IDR algorithm (not for the subroutine); changing the domain parameter
                         distribution is done by linear interpolation over this many iterations
        :param performance_threshold: lower bound for the performance that has to be reached until the domain parameter
                                      randomization is changed
        :param param_adjustment_portion: what portion of the IDR iterations should be spent on adjusting the domain
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

        self._subroutine = subroutine
        # Wrap the sampler with a rollout saving wrapper for the stopping criterion.
        self._subroutine.sampler = RolloutSavingWrapper(self._subroutine.sampler)
        self._subroutine.save_name = self._subroutine.name
        self._subroutine.stopping_criterion = self._subroutine.stopping_criterion | MinReturnStoppingCriterion(
            return_threshold=performance_threshold
        )

        self._env = env
        self._performance_threshold = performance_threshold
        self._param_adjustment_scale = param_adjustment_portion * max_iter

        self._parameters = [param for param in env.randomizer.domain_params if isinstance(param, SelfPacedDomainParam)]

    @property
    def sub_algorithm(self) -> Algorithm:
        """Get the policy optimization subroutine."""
        return self._subroutine

    @property
    def sample_count(self) -> int:
        # Forward to subroutine.
        return self._subroutine.sample_count

    def step(self, snapshot_mode: str, meta_info: dict = None):
        """
        Perform a step of IDR. This includes training the subroutine and updating the context distribution accordingly.
        For a description of the parameters see `pyrado.algorithms.base.Algorithm.step`.
        """
        self.save_snapshot()

        for param in self._parameters:
            self.logger.add_value(f"cur context mean for {param.name}", param.context_mean.item())
            self.logger.add_value(f"cur context cov for {param.name}", param.context_cov.item())

        self._subroutine.train(snapshot_mode, None, meta_info)

        for param in self._parameters:
            param.adapt(
                "context_mean",
                param.context_mean + (param.target_mean - param.init_mean) / self._param_adjustment_scale,
            )
            param.adapt(
                "context_cov_chol_flat",
                param.context_cov_chol_flat
                + (param.target_cov_chol_flat - param.init_cov_chol_flat) / self._param_adjustment_scale,
            )

    def reset(self, seed: int = None):
        # Forward to subroutine.
        self._subroutine.reset(seed)

    def save_snapshot(self, meta_info: dict = None):
        super().save_snapshot(meta_info)

        if meta_info is None:
            # This algorithm instance is not a subroutine of another algorithm.
            self._subroutine.save_snapshot(meta_info)
