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
import scipy as sp
import scipy.stats

import pyrado
from pyrado.algorithms.stopping_criteria.stopping_criterion import StoppingCriterion
from pyrado.algorithms.utils import RolloutSavingWrapper
from pyrado.sampling.sampler import SamplerBase


class RolloutBasedStoppingCriterion(StoppingCriterion):
    """
    Abstract extension of the base `StoppingCriterion` class for criteria that are based on having access to rollouts.

    .. note::
        Requires the algorithm to expose a `RolloutSavingWrapper` via a property `sampler`.
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
    """
    Abstract extension of the base `RolloutBasedStoppingCriterion` class for criteria that are based on a specific
    statistic of the returns of rollouts of the last iteration.
    """

    # List of the statistics that this class can compute from a rollout.
    AVAILABLE_RETURN_STATISTICS = ("min", "max", "median", "mean", "variance")

    def __init__(self, return_statistic="median", num_lookbacks=1):
        """
        Constructor.

        :param return_statistic: statistic to compute; must be one of `min`, `max`, `median`, `mean`, or `variance`
        :param num_lookbacks: over how many iterations the statistic should be computed; for example, a value of two
                              means that the rollouts of both the current and the previous iteration will be used for
                              computing the statistic; defaults to one
        """
        super().__init__()
        return_statistic = return_statistic.lower()
        if not (return_statistic in ReturnStatisticBasedStoppingCriterion.AVAILABLE_RETURN_STATISTICS):
            raise pyrado.ValueErr(
                msg=f"return_statistic has to be one of {ReturnStatisticBasedStoppingCriterion.AVAILABLE_RETURN_STATISTICS} (case insensitive)"
            )
        self._return_statistic = return_statistic
        self._num_lookbacks = num_lookbacks

    def _is_met_with_sampler(self, algo, sampler: RolloutSavingWrapper) -> bool:
        """
        Computes the return statistic if enough iterations have passed and forwards the computed statistic to the
        method `_is_met_with_return_statistic`.

        :param algo: instance of `Algorithm` that has to be evaluated
        :param sampler: instance of `RolloutSavingWrapper`, the sampler of `algo`, that has to be evaluated
        :return: `True` if the criterion is met, and `False` otherwise
        """
        if len(sampler.rollouts) < self._num_lookbacks:
            return False
        step_sequences = sampler.rollouts[-self._num_lookbacks :]
        returns = [rollout.undiscounted_return() for step_sequence in step_sequences for rollout in step_sequence]
        return_statistic = self._compute_return_statistic(np.asarray(returns))
        return self._is_met_with_return_statistic(algo, sampler, return_statistic)

    @abstractmethod
    def _is_met_with_return_statistic(self, algo, sampler: RolloutSavingWrapper, return_statistic: float) -> bool:
        """
        Checks whether the stopping criterion is met.

        .. note::
            Has to be overwritten by sub-classes.

        :param algo: instance of `Algorithm` that has to be evaluated
        :param sampler: instance of `RolloutSavingWrapper`, the sampler of `algo`, that has to be evaluated
        :param return_statistic: statistic that has been computed for the latest rollouts
        :return: `True` if the criterion is met, and `False` otherwise
        """
        raise NotImplementedError()

    def _compute_return_statistic(self, returns: np.ndarray) -> float:
        """
        Computes the desired statistic of the given list of returns according to the statistic requested in the
        constructor.

        :param returns: returns
        :return: statistic
        """
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

    def _is_met_with_return_statistic(self, algo, sampler: RolloutSavingWrapper, return_statistic: float) -> bool:
        """Returns whether the return statistic is greater than or equal to the return threshold."""
        return return_statistic >= self._return_threshold


class ConvergenceStoppingCriterion(ReturnStatisticBasedStoppingCriterion):
    """
    Checks for convergence of the returns for a given statistic that can be specified in the constructor. This is done
    by fitting a linear regression model to all the previous statistics (stored in a list) and performing a Wald test
    with a t-distribution of the test statistic (with tne null hypothesis that the slope is zero). The resulting
    p-value is called the *probability of convergence* and is used for checking if the algorithm has converged.

    This procedure can intuitively be explained by measuring "how flat the returns are" in the presence of noise. It has
    the advantage over just checking how much the return changes that it is independent of the noise on the returns,
    i.e. no specific threshold has to be hand-tuned.

    This criterion has to modes: moving and cumulative. In the moving mode, only the latest `M` values are used for
    fitting the linear model, and in the first `M - 1` iterations the criterion is treated as not being met. In the
    cumulative mode, all the previous values are used and only the first iteration is treated as not being met as there
    have to be at least two points to fit a linear model. While the former is primarily useful for convergence checking
    for a regular algorithm, the latter is primarily useful for checking convergence of the subroutine in a
    meta-algorithm as here it is possible that convergence kicks in far at the beginning of the learning process as the
    environment did not change much (see, for example, SPRL).

    It might be helpful to combine this stopping criterion with a min-iterations criterion (TODO) to ensure that the
    algorithm does not terminate prematurely due to initialization issues. For example, PPO usually takes some
    iterations to make progress which leads to a flat learning curve that however does not correspond to the algorithm
    being converged.
    """

    # TODO: Stopped here. Next steps: Implement the min-iterations criterion and the two modes (moving and cumulative).

    def __init__(self, convergence_probability_threshold=0.99, return_statistic="median", num_lookbacks=1):
        """
        Constructor.

        :param convergence_probability_threshold: threshold of the p-value above which the algorithm is considered to be
                                                  converged; defaults to `0.99`, i.e. a `99%` certainty that the data
                                                  can be explained
        """
        super().__init__(return_statistic, num_lookbacks)
        self._convergence_probability_threshold = convergence_probability_threshold
        self._return_statistic_history = []

    def __repr__(self) -> str:
        return (
            f"ConvergenceStoppingCriterion[return_statistic={self._return_statistic}, "
            f"num_lookbacks={self._num_lookbacks}, "
            f"return_statistic_history={self._return_statistic_history}]"
        )

    def __str__(self) -> str:
        return f"({self._return_statistic} return converged)"

    def _reset(self) -> NoReturn:
        self._return_statistic_history = []

    def _is_met_with_return_statistic(self, algo, sampler: RolloutSavingWrapper, return_statistic: float) -> bool:
        self._return_statistic_history.append(return_statistic)
        convergence_prob = self._compute_convergence_probability()
        return convergence_prob >= self._convergence_probability_threshold

    def _compute_convergence_probability(self) -> float:
        return sp.stats.linregress(range(len(self._return_statistic_history)), self._return_statistic_history).pvalue
