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

from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.spaces.box import BoxSpace


@pytest.mark.wrapper
def test_no_downsampling():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=1)

    # Perform some actions
    wenv.step(np.array([4, 1]))
    assert mockenv.last_act == [4, 1]
    wenv.step(np.array([7, 5]))
    assert mockenv.last_act == [7, 5]


@pytest.mark.wrapper
def test_act_downsampling():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=2)

    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([2, 4]))  # should be ignored
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [1, 2]
    wenv.step(np.array([2, 3]))  # should be ignored
    assert mockenv.last_act == [1, 2]


@pytest.mark.wrapper
def test_reset():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=2)

    # Perform some actions
    wenv.step(np.array([0, 4]))
    assert mockenv.last_act == [0, 4]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [0, 4]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [4, 4]

    # The next action would be [4, 4] again, but now we reset
    wenv.reset()
    assert wenv._act_last is None
    assert wenv._cnt == 0

    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [1, 2]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [1, 2]


@pytest.mark.wrapper
def test_domain_param():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv = DownsamplingWrapper(mockenv, factor=2)

    # Reset to initialize buffer
    wenv.reset()

    # Perform some actions
    wenv.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([2, 4]))
    assert mockenv.last_act == [0, 1]
    wenv.step(np.array([4, 4]))
    assert mockenv.last_act == [4, 4]

    # change the downsampling and reset
    wenv.domain_param = {"downsampling": 1}
    wenv.reset()

    wenv.step(np.array([1, 2]))
    assert mockenv.last_act == [1, 2]
    wenv.step(np.array([2, 3]))
    assert mockenv.last_act == [2, 3]
    wenv.step(np.array([8, 9]))
    assert mockenv.last_act == [8, 9]


@pytest.mark.wrapper
def test_combination_downsampling_delay():
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv_ds_dl = DownsamplingWrapper(mockenv, factor=2)
    wenv_ds_dl = ActDelayWrapper(wenv_ds_dl, delay=3)

    # Reset to initialize buffer
    wenv_ds_dl.reset()

    # The first ones are 0 because the ActDelayWrapper's queue is initialized with 0
    wenv_ds_dl.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 2]))
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 3]))
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 4]))
    # Intuitively one would think last_act would be [0, 1] here, but this is the effect of the wrappers' combination
    assert mockenv.last_act == [0, 0]
    wenv_ds_dl.step(np.array([0, 5]))
    assert mockenv.last_act == [0, 2]
    wenv_ds_dl.step(np.array([0, 6]))
    assert mockenv.last_act == [0, 2]
    wenv_ds_dl.step(np.array([0, 7]))
    assert mockenv.last_act == [0, 4]
    wenv_ds_dl.step(np.array([0, 8]))
    assert mockenv.last_act == [0, 4]
    wenv_ds_dl.step(np.array([0, 9]))
    assert mockenv.last_act == [0, 6]
    wenv_ds_dl.step(np.array([1, 0]))
    assert mockenv.last_act == [0, 6]


@pytest.mark.wrapper
def test_combination_delay_downsampling():
    """After delay number of actions, the actions are downsampled by the factor"""
    mockenv = MockEnv(act_space=BoxSpace(-1, 1, shape=(2,)), obs_space=BoxSpace(-1, 1, shape=(2,)))
    wenv_dl_ds = ActDelayWrapper(mockenv, delay=3)
    wenv_dl_ds = DownsamplingWrapper(wenv_dl_ds, factor=2)

    # Reset to initialize buffer
    wenv_dl_ds.reset()

    # The first ones are 0 because the ActDelayWrapper's queue is initialized with 0
    wenv_dl_ds.step(np.array([0, 1]))
    assert mockenv.last_act == [0, 0]
    wenv_dl_ds.step(np.array([0, 2]))
    assert mockenv.last_act == [0, 0]
    wenv_dl_ds.step(np.array([0, 3]))
    assert mockenv.last_act == [0, 0]
    # One time step earlier than the other order of wrappers
    wenv_dl_ds.step(np.array([0, 4]))
    assert mockenv.last_act == [0, 1]
    wenv_dl_ds.step(np.array([0, 5]))
    assert mockenv.last_act == [0, 1]
    wenv_dl_ds.step(np.array([0, 6]))
    assert mockenv.last_act == [0, 3]
    wenv_dl_ds.step(np.array([0, 7]))
    assert mockenv.last_act == [0, 3]
    wenv_dl_ds.step(np.array([0, 8]))
    assert mockenv.last_act == [0, 5]
    wenv_dl_ds.step(np.array([0, 9]))
    assert mockenv.last_act == [0, 5]
    wenv_dl_ds.step(np.array([1, 0]))
    assert mockenv.last_act == [0, 7]
    wenv_dl_ds.step(np.array([1, 1]))
    assert mockenv.last_act == [0, 7]
