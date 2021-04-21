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
from tests.environment_wrappers.mock_env import MockEnv

from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.spaces.box import BoxSpace


@pytest.mark.wrapper
def test_space():
    # Use mock env
    mockenv = MockEnv(act_space=BoxSpace([-2, -1, 0], [2, 3, 1]))

    uut = ActNormWrapper(mockenv)

    # Check action space bounds
    lb, ub = uut.act_space.bounds
    assert np.all(lb == -1)
    assert np.all(ub == 1)


@pytest.mark.wrapper
def test_denormalization():
    # Use mock env
    mockenv = MockEnv(act_space=BoxSpace([-2, -1, 0], [2, 3, 1]))

    uut = ActNormWrapper(mockenv)

    # Pass a bunch of actions
    uut.step(np.array([0, 0, 0]))
    assert mockenv.last_act == [0, 1, 0.5]

    uut.step(np.array([1, 1, 1]))
    assert mockenv.last_act == [2, 3, 1]

    uut.step(np.array([-1, -1, -1]))
    assert mockenv.last_act == [-2, -1, 0]
