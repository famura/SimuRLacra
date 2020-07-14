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

from pyrado.spaces.box import BoxSpace
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from tests.environment_wrappers.mock_env import MockEnv


@pytest.mark.wrappers
def test_spaces():
    mockenv = MockEnv(obs_space=BoxSpace([-1, -2, -3], [1, 2, 3], labels=['one', 'two', 'three']))

    # Use a simple mask to drop the second element
    mask = [0, 1, 0]
    wenv = ObsPartialWrapper(mockenv, mask)

    # Check resulting space
    lb, ub = wenv.obs_space.bounds
    assert list(lb) == [-1, -3]
    assert list(ub) == [1, 3]
    assert list(wenv.obs_space.labels) == ['one', 'three']


@pytest.mark.wrappers
def test_values():
    mockenv = MockEnv(obs_space=BoxSpace([-1, -2, -3], [1, 2, 3], labels=['one', 'two', 'three']))

    # Use a simple mask to drop the second element
    mask = [0, 1, 0]
    wenv = ObsPartialWrapper(mockenv, mask)

    # Test some observation values
    mockenv.next_obs = [1, 2, 3]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [1, 3]

    mockenv.next_obs = [4, 7, 9]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [4, 9]


@pytest.mark.wrappers
def test_mask_invert():
    mockenv = MockEnv(obs_space=BoxSpace([-1, -2, -3], [1, 2, 3], labels=['one', 'two', 'three']))

    # Use a simple mask to drop the second element
    mask = [0, 1, 0]
    wenv = ObsPartialWrapper(mockenv, mask, keep_selected=True)

    # Test some observation values
    mockenv.next_obs = [1, 2, 3]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [2]

    mockenv.next_obs = [4, 7, 9]
    obs, _, _, _ = wenv.step(None)
    assert list(obs) == [7]


@pytest.mark.wrappers
def test_mask_from_indices():
    # Test the create_mask helper separately
    space = BoxSpace(-1, 1, shape=5)
    indices = [1, 4]

    mask = space.create_mask(indices)
    assert list(mask) == [0, 1, 0, 0, 1]


@pytest.mark.wrappers
def test_mask_from_labels():
    # Test the create_mask helper separately
    space = BoxSpace(-1, 1, shape=5, labels=['w', 'o', 'r', 'l', 'd'])
    indices = ['w', 'o']

    mask = space.create_mask(indices)
    assert list(mask) == [1, 1, 0, 0, 0]
