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
from copy import deepcopy
from types import SimpleNamespace
from typing import List, Optional

import numpy as np
import pytest

import pyrado
from pyrado.algorithms.stopping_criteria.predefined_criteria import (
    AlwaysStopStoppingCriterion,
    CustomStoppingCriterion,
    IterCountStoppingCriterion,
    NeverStopStoppingCriterion,
    SampleCountStoppingCriterion,
)
from pyrado.algorithms.stopping_criteria.rollout_based_criteria import (
    MinReturnStoppingCriterion,
    ReturnStatisticBasedStoppingCriterion,
)
from pyrado.algorithms.stopping_criteria.stopping_criterion import _AndStoppingCriterion, _OrStoppingCriterion
from pyrado.algorithms.utils import RolloutSavingWrapper
from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.sampling.sampler import SamplerBase
from pyrado.sampling.step_sequence import StepSequence


class MockSampler(SamplerBase):
    def __init__(self, step_sequences: List[StepSequence]):
        super().__init__(min_rollouts=0, min_steps=0)

        self._step_sequences = step_sequences

    def reinit(self, env: Optional[Env] = None, policy: Optional[Policy] = None):
        pass

    def sample(self) -> List[StepSequence]:
        return deepcopy(self._step_sequences)


class ExposingReturnStatisticBasedStoppingCriterion(ReturnStatisticBasedStoppingCriterion):
    def __init__(self, return_statistic="median", num_lookbacks=1):
        super().__init__(return_statistic, num_lookbacks)
        self.return_statistic_value = np.nan

    def _is_met_with_return_statistic(self, algo, sampler: RolloutSavingWrapper, return_statistic: float) -> bool:
        self.return_statistic_value = return_statistic
        return False


# noinspection PyTypeChecker
def test_magic_function_implementation_and():
    a = CustomStoppingCriterion(None, "A")
    b = CustomStoppingCriterion(None, "B")
    for criterion, expected_str in [
        (a & a, "(A and A)"),
        (b & b, "(B and B)"),
        (a & b, "(A and B)"),
        (b & a, "(B and A)"),
    ]:
        assert isinstance(criterion, _AndStoppingCriterion)
        assert str(criterion) == expected_str


# noinspection PyTypeChecker
def test_magic_function_implementation_or():
    a = CustomStoppingCriterion(None, "A")
    b = CustomStoppingCriterion(None, "B")
    for criterion, expected_str in [(a | a, "(A or A)"), (b | b, "(B or B)"), (a | b, "(A or B)"), (b | a, "(B or A)")]:
        assert isinstance(criterion, _OrStoppingCriterion)
        assert str(criterion) == expected_str


def test_criterion_combination_and():
    a = AlwaysStopStoppingCriterion()
    b = NeverStopStoppingCriterion()
    a_and_a = _AndStoppingCriterion(a, a)
    b_and_b = _AndStoppingCriterion(b, b)
    a_and_b = _AndStoppingCriterion(a, b)
    b_and_a = _AndStoppingCriterion(b, a)

    assert str(a_and_a) == "(True and True)"
    assert str(b_and_b) == "(False and False)"
    assert str(a_and_b) == "(True and False)"
    assert str(b_and_a) == "(False and True)"
    assert a_and_a.is_met(None)
    assert not b_and_b.is_met(None)
    assert not a_and_b.is_met(None)
    assert not b_and_a.is_met(None)


def test_criterion_combination_or():
    a = AlwaysStopStoppingCriterion()
    b = NeverStopStoppingCriterion()
    a_or_a = _OrStoppingCriterion(a, a)
    b_or_b = _OrStoppingCriterion(b, b)
    a_or_b = _OrStoppingCriterion(a, b)
    b_or_a = _OrStoppingCriterion(b, a)

    assert str(a_or_a) == "(True or True)"
    assert str(b_or_b) == "(False or False)"
    assert str(a_or_b) == "(True or False)"
    assert str(b_or_a) == "(False or True)"
    assert a_or_a.is_met(None)
    assert not b_or_b.is_met(None)
    assert a_or_b.is_met(None)
    assert b_or_a.is_met(None)


def test_criterion_always():
    a = AlwaysStopStoppingCriterion()

    assert a.is_met(None)


def test_criterion_never():
    a = NeverStopStoppingCriterion()

    assert not a.is_met(None)


@pytest.mark.parametrize("is_met_expected", [True, False])
def test_criterion_custom(is_met_expected):
    # Assigning to a variable in a closure would redefine the scope, so rather use a list as a holding.
    was_called = [False]
    algo_expected = "ABC"

    def criterion_fn(algo):
        was_called[0] = True
        assert algo == algo_expected
        return is_met_expected

    criterion = CustomStoppingCriterion(criterion_fn, "Name")

    assert str(criterion) == "Name"
    assert criterion.is_met(algo_expected) == is_met_expected
    assert was_called[0]


def test_criterion_iter_count_lower():
    algo = SimpleNamespace(curr_iter=1)
    criterion = IterCountStoppingCriterion(max_iter=2)
    assert not criterion.is_met(algo)


def test_criterion_iter_count_higher():
    algo = SimpleNamespace(curr_iter=3)
    criterion = IterCountStoppingCriterion(max_iter=2)
    assert criterion.is_met(algo)


def test_criterion_iter_count_equal():
    algo = SimpleNamespace(curr_iter=2)
    criterion = IterCountStoppingCriterion(max_iter=2)
    assert criterion.is_met(algo)


def test_criterion_sample_count_lower():
    algo = SimpleNamespace(sample_count=1)
    criterion = SampleCountStoppingCriterion(max_sample_count=2)
    assert not criterion.is_met(algo)


def test_criterion_sample_count_higher():
    algo = SimpleNamespace(sample_count=3)
    criterion = SampleCountStoppingCriterion(max_sample_count=2)
    assert criterion.is_met(algo)


def test_criterion_sample_count_equal():
    algo = SimpleNamespace(sample_count=2)
    criterion = SampleCountStoppingCriterion(max_sample_count=2)
    assert criterion.is_met(algo)


# noinspection PyTypeChecker
def test_criterion_rollout_based_no_sampler():
    algo = SimpleNamespace()
    criterion = MinReturnStoppingCriterion(return_threshold=None)
    with pytest.raises(pyrado.ValueErr):
        criterion.is_met(algo)


# noinspection PyTypeChecker
def test_criterion_rollout_based_wrong_sampler():
    sampler = SimpleNamespace()
    algo = SimpleNamespace(sampler=sampler)
    criterion = MinReturnStoppingCriterion(return_threshold=None)
    with pytest.raises(pyrado.TypeErr):
        criterion.is_met(algo)


# noinspection PyTypeChecker
@pytest.mark.parametrize(
    ["statistic", "expected"], [("min", 1), ("max", 6), ("median", 2), ("mean", 3), ("variance", 14 / 3)]
)
def test_criterion_return_statistic_based_check_min(statistic, expected):
    rollout_a = SimpleNamespace(undiscounted_return=lambda: 6)
    rollout_b = SimpleNamespace(undiscounted_return=lambda: 2)
    rollout_c = SimpleNamespace(undiscounted_return=lambda: 1)
    sampler = RolloutSavingWrapper(MockSampler([rollout_a, rollout_b, rollout_c]))
    sampler.sample()
    algo = SimpleNamespace(sampler=sampler)
    criterion = ExposingReturnStatisticBasedStoppingCriterion(return_statistic=statistic)
    criterion.is_met(algo)
    assert np.isclose(criterion.return_statistic_value, expected)


# noinspection PyTypeChecker
def test_criterion_rollout_based_min_min_return_lower():
    rollout_a = SimpleNamespace(undiscounted_return=lambda: 1)
    sampler = RolloutSavingWrapper(MockSampler([rollout_a]))
    sampler.sample()
    algo = SimpleNamespace(sampler=sampler)
    criterion = MinReturnStoppingCriterion(return_threshold=2)
    assert not criterion.is_met(algo)


# noinspection PyTypeChecker
def test_criterion_rollout_based_min_min_return_higher():
    rollout_a = SimpleNamespace(undiscounted_return=lambda: 3)
    sampler = RolloutSavingWrapper(MockSampler([rollout_a]))
    sampler.sample()
    algo = SimpleNamespace(sampler=sampler)
    criterion = MinReturnStoppingCriterion(return_threshold=2)
    assert criterion.is_met(algo)


# noinspection PyTypeChecker
def test_criterion_rollout_based_min_min_return_equal():
    rollout_a = SimpleNamespace(undiscounted_return=lambda: 2)
    sampler = RolloutSavingWrapper(MockSampler([rollout_a]))
    sampler.sample()
    algo = SimpleNamespace(sampler=sampler)
    criterion = MinReturnStoppingCriterion(return_threshold=2)
    assert criterion.is_met(algo)
