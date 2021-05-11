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

from abc import ABC, abstractmethod
from typing import NoReturn


class StoppingCriterion(ABC):
    """
    Base class for the stopping criterion. A stopping criterion takes an algorithm (and hence its current state) and
    decides whether the algorithm should terminate. A common stopping criterion is e.g. reaching a set number of
    iterations.
    """

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        """Returns an `_OrStoppingCriterion` that combines this criterion with the other."""
        return _OrStoppingCriterion(self, other)

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        """Returns an `_AndStoppingCriterion` that combines this criterion with the other."""
        return _AndStoppingCriterion(self, other)

    @abstractmethod
    def is_met(self, algo) -> bool:
        """
        Checks whether the stopping criterion is met.

        .. note::
            Has to be overwritten by sub-classes.

        :param algo: instance of `Algorithm` that has to be evaluated
        :return: `True` if the criterion is met, and `False` otherwise
        """
        raise NotImplementedError()

    def reset(self) -> NoReturn:
        """Resets the internal state of this stopping criterion. Has to be called when the algorithm is reset."""
        pass


class _AndStoppingCriterion(StoppingCriterion):
    """Combines two stopping criteria such that both of them have to be met in order for this criterion to be met."""

    def __init__(self, criterion1: StoppingCriterion, criterion2: StoppingCriterion):
        """
        Constructor.

        :param criterion1: first criterion; gets evaluated first
        :param criterion2: second criterion
        """
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def __repr__(self) -> str:
        return f"_AndStoppingCriterion[criterion1={repr(self.criterion1)}; criterion2={repr(self.criterion2)}]"

    def __str__(self) -> str:
        return f"({self.criterion1} and {self.criterion2})"

    def is_met(self, algo) -> bool:
        """Checks if both criteria are met."""
        return self.criterion1.is_met(algo) and self.criterion2.is_met(algo)


class _OrStoppingCriterion(StoppingCriterion):
    """Combines two stopping criteria such that at least one of them has to be met in order for this criterion to be met."""

    def __init__(self, criterion1: StoppingCriterion, criterion2: StoppingCriterion):
        """
        Constructor.

        :param criterion1: first criterion; gets evaluated first
        :param criterion2: second criterion
        """
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def __repr__(self) -> str:
        return f"_OrStoppingCriterion[criterion1={repr(self.criterion1)}; criterion2={repr(self.criterion2)}]"

    def __str__(self) -> str:
        return f"({self.criterion1} or {self.criterion2})"

    def is_met(self, algo) -> bool:
        """Checks if at least one criterion is met."""
        return self.criterion1.is_met(algo) or self.criterion2.is_met(algo)
