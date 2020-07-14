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
import pytest

from pyrado.spaces.box import BoxSpace
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from tests.environment_wrappers.mock_env import MockEnv


@pytest.mark.wrappers
def test_no_delay():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=0)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([4, 1]))
    assert mockenv.last_act == [4, 1]
    wenv.step(np.array([7, 5]))
    assert mockenv.last_act == [7, 5]


@pytest.mark.wrappers
def test_act_delay():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=2)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 4]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [2, 4]


@pytest.mark.wrappers
def test_reset():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=1)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0, 4]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [0, 4]

    # The next action would be [4, 4], but now we reset again
    wenv.reset()

    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [1, 2]


@pytest.mark.wrappers
def test_domain_param():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = ActDelayWrapper(mockenv, delay=1)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 4]))
    assert mockenv.last_act == [0, 1]

    # change the delay and reset
    wenv.domain_param = {'act_delay': 2}
    wenv.reset()

    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [0, 0]
    wenv.step(np.array([8, 9]))
    assert mockenv.last_act == [1, 2]
