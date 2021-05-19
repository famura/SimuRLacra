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

from abc import abstractmethod
from typing import NoReturn, Optional

import numpy as np

import pyrado
from pyrado.algorithms.stopping_criteria.stopping_criterion import StoppingCriterion
from pyrado.algorithms.utils import RolloutSavingWrapper
from pyrado.sampling.sampler import SamplerBase


class RolloutBasedStoppingCriterion(StoppingCriterion):
    """
    Abstract extension of the base `StoppingCriterion` class for criteria that are based on having access to rollouts.
    This criterion requires the algorithm to expose a `RolloutSavingWrapper` via a property `sampler`.
    """

    def is_met(self, algo) -> bool:
        """
        Gets the sampler from the algorithm, checks if it is a `RolloutSavingWrapper` and forwards the checkinf of the
        stopping criterion to `_is_met_with_sampler(..)`.

        :param algo: instance of `Algorithm` that has to be evaluated
        :return: `True` if the criterion is met, and `False` otherwise
        """
        if not hasattr(algo, "sampler"):
            raise pyrado.ValueErr(
                msg="Any rollout-based stopping criterion requires the algorithm to expose a property 'sampler'!"
            )
        sampler: Optional[SamplerBase] = algo.sampler
        if not isinstance(sampler, RolloutSavingWrapper):
            raise pyrado.TypeErr(
                msg="Any rollout-based stopping criterion requires the algorithm to expose a sampler of type 'RolloutSavingWrapper' via the property 'sampler'!"
            )
        return self._is_met_with_sampler(algo, sampler)

    @abstractmethod
    def _is_met_with_sampler(self, algo, sampler: RolloutSavingWrapper) -> bool:
        """
        Checks whether the stopping criterion is met.

        .. note::
            Has to be overwritten by sub-classes.

        :param algo: instance of `Algorithm` that has to be evaluated
        :param sampler: instance of `RolloutSavingWrapper`, the sampler of `algo`, that has to be evaluated
        :return: `True` if the criterion is met, and `False` otherwise
        """
        raise NotImplementedError()


class ReturnStatisticBasedStoppingCriterion(RolloutBasedStoppingCriterion):
    AVAILABLE_RETURN_STATISTICS = ("min", "max", "median", "mean", "variance")

    def __init__(self, return_statistic="median", num_lookbacks=1):
        return_statistic = return_statistic.lower()
        if not (return_statistic in ReturnStatisticBasedStoppingCriterion.AVAILABLE_RETURN_STATISTICS):
            raise pyrado.ValueErr(
                msg=f"return_statistic has to be one of {ReturnStatisticBasedStoppingCriterion.AVAILABLE_RETURN_STATISTICS} (case insensitive)"
            )
        self._return_statistic = return_statistic
        self._num_lookbacks = num_lookbacks

    def _is_met_with_sampler(self, algo, sampler: RolloutSavingWrapper) -> bool:
        if len(sampler.rollouts) < self._num_lookbacks:
            return False
        step_sequences = sampler.rollouts[-self._num_lookbacks :]
        returns = [rollout.undiscounted_return() for step_sequence in step_sequences for rollout in step_sequence]
        return_statistic = self._compute_return_statistic(np.asarray(returns))
        return self._is_met_with_return_statistic(algo, sampler, return_statistic)

    @abstractmethod
    def _is_met_with_return_statistic(self, algo, sampler: RolloutSavingWrapper, return_statistic: float) -> bool:
        raise NotImplementedError()

    def _compute_return_statistic(self, returns: np.ndarray) -> float:
        if self._return_statistic == "min":
            return np.min(returns)
        if self._return_statistic == "max":
            return np.max(returns)
        if self._return_statistic == "median":
            return np.quantile(returns, q=0.50)
        if self._return_statistic == "mean":
            return np.mean(returns).item()
        if self._return_statistic == "variance":
            return returns.var().item()
        assert (
            False
        ), "Should not happen! Either the code is inconsistent or the instance variable _return_statistic has been touched!"


class MinReturnStoppingCriterion(ReturnStatisticBasedStoppingCriterion):
    """
    Uses any statistic (defaulting to min) of the return of the latest rollout as a stopping criterion and stops if this
    statistic exceeds a certain threshold.
    """

    def __init__(self, return_threshold: float, return_statistic="min"):
        """
        Constructor.

        :param return_threshold: return threshold; if the return statistic reaches this threshold, the stopping
                                 criterion is met
        :param return_statistic: the statistic of the return to use; defaults to minimum
        """
        super().__init__(return_statistic=return_statistic)
        self._return_threshold = return_threshold

    def __repr__(self) -> str:
        return f"MinReturnStoppingCriterion[return_statistic={self._return_statistic}, min_return={self._return_threshold}]"

    def __str__(self) -> str:
        return f"({self._return_statistic} return >= {self._return_threshold})"

    # noinspection PyUnusedLocal
    def _is_met_with_return_statistic(self, algo, sampler: RolloutSavingWrapper, return_statistic: float) -> bool:
        """Returns whether the return statistic is greater than or equal to the return threshold."""
        return return_statistic >= self._return_threshold


class ConvergenceStoppingCriterion(ReturnStatisticBasedStoppingCriterion):
    """Uses the minimum return of the latest rollout as a stopping criterion."""

    def __init__(self, return_statistic="median", num_lookbacks=1):
        super().__init__(return_statistic, num_lookbacks)
        self._return_statistic_history = []

    def __repr__(self) -> str:
        return (
            f"ConvergenceStoppingCriterion[return_statistic={self._return_statistic}, "
            f"num_lookbacks={self._num_lookbacks}, "
            f"return_statistic_history={self._return_statistic_history}]"
        )

    def __str__(self) -> str:
        return f"({self._return_statistic} return converged)"

    def reset(self) -> NoReturn:
        self._return_statistic_history = []

    # noinspection PyUnusedLocal
    def _is_met_with_return_statistic(self, algo, sampler: RolloutSavingWrapper, return_statistic: float) -> bool:
        pass
