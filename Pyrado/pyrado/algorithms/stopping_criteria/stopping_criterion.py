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
    def __init__(self):
        self._criterion = self

    def __call__(self) -> bool:
        return self._validate()

    def __and__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        return _AndStoppingCriterion(self._criterion, other)

    def __or__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        return _OrStoppingCriterion(self._criterion, other)

    def __xor__(self, other: "StoppingCriterion") -> "StoppingCriterion":
        return _XorStoppingCriterion(self._criterion, other)

    def __iand__(self, other: "StoppingCriterion") -> NoReturn:
        self._criterion = self._criterion & other

    def __ior__(self, other: "StoppingCriterion") -> NoReturn:
        self._criterion = self._criterion | other

    def __ixor__(self, other: "StoppingCriterion") -> NoReturn:
        self._criterion = self._criterion ^ other

    @abstractmethod
    def _validate(self) -> bool:
        raise NotImplementedError()


class _AndStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion1: StoppingCriterion, criterion2: StoppingCriterion):
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def _validate(self) -> bool:
        return self.criterion1() and self.criterion2()


class _OrStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion1: StoppingCriterion, criterion2: StoppingCriterion):
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def _validate(self) -> bool:
        return self.criterion1() or self.criterion2()


class _XorStoppingCriterion(StoppingCriterion):
    def __init__(self, criterion1: StoppingCriterion, criterion2: StoppingCriterion):
        super().__init__()
        self.criterion1 = criterion1
        self.criterion2 = criterion2

    def _validate(self) -> bool:
        return self.criterion1() == self.criterion2()
