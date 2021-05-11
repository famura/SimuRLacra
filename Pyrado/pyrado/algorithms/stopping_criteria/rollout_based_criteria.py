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
from typing import Optional

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


class MinReturnStoppingCriterion(RolloutBasedStoppingCriterion):
    """Uses the minimum return of the latest rollout as a stopping criterion."""

    def __init__(self, min_return: float):
        """
        Constructor.

        :param min_return: minimal return; if this return is reached, the stopping criterion is met
        """
        self._min_return = min_return

    def __repr__(self) -> str:
        return f"MinReturnStoppingCriterion[min_return={self._min_return}]"

    def __str__(self) -> str:
        return f"(return >= {self._min_return})"

    # noinspection PyUnusedLocal
    def _is_met_with_sampler(self, algo, sampler: RolloutSavingWrapper) -> bool:
        """Returns whether the minimum return of the latest rollout is greater than or equal to the minimum return."""
        rollouts = sampler.rollouts[-1]
        returns = [rollout.undiscounted_return() for rollout in rollouts]
        min_return = np.min(returns)
        return min_return >= self._min_return
