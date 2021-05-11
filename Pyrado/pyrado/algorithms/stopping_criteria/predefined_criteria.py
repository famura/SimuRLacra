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

from typing import Any, Callable, Optional

from pyrado.algorithms.stopping_criteria.stopping_criterion import StoppingCriterion


class AlwaysStopStoppingCriterion(StoppingCriterion):
    """Stopping criterion that is always met."""

    def __repr__(self) -> str:
        return "AlwaysStopStoppingCriterion"

    def __str__(self) -> str:
        return "True"

    def is_met(self, algo) -> bool:
        """Returns `True`."""
        return True


class NeverStopStoppingCriterion(StoppingCriterion):
    """Stopping criterion that is never met."""

    def __repr__(self) -> str:
        return "NeverStopStoppingCriterion"

    def __str__(self) -> str:
        return "False"

    def is_met(self, algo) -> bool:
        """Returns `False`."""
        return False


class CustomStoppingCriterion(StoppingCriterion):
    """Custom stopping criterion that takes an arbitrary callable to evaluate."""

    def __init__(self, criterion_fn: Callable[[Any], bool], name: Optional[str] = None):
        """
        Constructor.

        :param criterion_fn: signature `[Algorithm] -> bool`; gets evaluated when `is_met` is called; allows for custom
                             functionality, e.g. if an algorithm requires special treatment; the given algorithm is the
                             same that was passed to the `is_met` method
        :param name: name of the stopping criterion, used for `str(..)` and ´repr(..)`
        """
        self._criterion_fn = criterion_fn
        self._name = name

    def __repr__(self) -> str:
        return f"CustomStoppingCriterion[_criterion_fn={repr(self._criterion_fn)}; name={self._name}]"

    def __str__(self) -> str:
        return "Custom" if self._name is None else self._name

    def is_met(self, algo) -> bool:
        """Invokes the criterion function that was passed to the constructor."""
        return self._criterion_fn(algo)


class IterCountStoppingCriterion(StoppingCriterion):
    """Uses the iteration number as a stopping criterion, i.e. sets a maximum number of iterations."""

    def __init__(self, max_iter: int):
        """
        Constructor.

        :param max_iter: maximum number of iterations
        """
        self._max_iter = max_iter

    def is_met(self, algo) -> bool:
        """Returns whether the current iteration number os greater than or equal to the maximum number of iterations."""
        return algo.curr_iter >= self._max_iter


class SampleCountStoppingCriterion(StoppingCriterion):
    """Uses the sampler count as a stopping criterion, i.e. sets a maximum number samples."""

    def __init__(self, max_sample_count: int):
        """
        Constructor.

        :param max_sample_count: maximum sample count
        """
        self._max_sample_count = max_sample_count

    def is_met(self, algo) -> bool:
        """Returns whether the current sample count is greater than or equal to the maximum sample count."""
        return algo.sample_count >= self._max_sample_count
