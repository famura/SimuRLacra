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

import pytest
import numpy as np

from pyrado.environment_wrappers.action_discrete import ActDiscreteWrapper
from pyrado.spaces.box import BoxSpace
from pyrado.spaces.discrete import DiscreteSpace
from tests.environment_wrappers.mock_env import MockEnv


@pytest.mark.wrapper
@pytest.mark.parametrize("num_bins", list(range(1, 500 + 1)))
def test_action_space_eles(num_bins: int):
    mockenv = MockEnv(act_space=BoxSpace(-1.0, 1.0, shape=(1,)))
    wenv = ActDiscreteWrapper(mockenv, num_bins=num_bins)

    # Reset to initialize buffer
    wenv.reset()

    # Test if action space is correct
    assert isinstance(wenv.act_space, DiscreteSpace)
    assert (wenv.act_space.eles == np.array(range(num_bins)).reshape((-1, 1))).all()


@pytest.mark.wrapper
def test_one_bin():
    mockenv = MockEnv(act_space=BoxSpace(-1.0, 1.0, shape=(1,)))
    wenv = ActDiscreteWrapper(mockenv, num_bins=1)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0]))
    assert mockenv.last_act == [-1.0]


@pytest.mark.wrapper
def test_two_bins():
    mockenv = MockEnv(act_space=BoxSpace(-1.0, 1.0, shape=(1,)))
    wenv = ActDiscreteWrapper(mockenv, num_bins=2)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0]))
    assert mockenv.last_act == [-1.0]
    wenv.step(np.array([1]))
    assert mockenv.last_act == [1.0]


@pytest.mark.wrapper
def test_five_bins():
    mockenv = MockEnv(act_space=BoxSpace(-1.0, 1.0, shape=(1,)))
    wenv = ActDiscreteWrapper(mockenv, num_bins=5)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0]))
    assert mockenv.last_act == [-1.0]
    wenv.step(np.array([1]))
    assert mockenv.last_act == [-0.5]
    wenv.step(np.array([2]))
    assert mockenv.last_act == [0.0]
    wenv.step(np.array([3]))
    assert mockenv.last_act == [0.5]
    wenv.step(np.array([4]))
    assert mockenv.last_act == [1.0]
