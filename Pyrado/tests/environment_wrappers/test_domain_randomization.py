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

from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer, DomainRandWrapperLive
from pyrado.environments.sim_base import SimEnv


@pytest.mark.wrapper
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qbb",
    ],
    indirect=True,
)
def test_dr_wrapper_live_bob(env):
    param_init = env.domain_param
    randomizer = create_default_randomizer(env)
    wrapper = DomainRandWrapperLive(env, randomizer)
    # So far no randomization happened, thus the parameter should be equal
    assert env.domain_param == param_init

    # Randomize 10 times 1 new parameter set
    for _ in range(10):
        param_old = wrapper.domain_param
        wrapper.reset()
        assert param_old != wrapper.domain_param


@pytest.mark.wrapper
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qbb",
    ],
    indirect=True,
)
@pytest.mark.parametrize("selection", ["cyclic", "random"])
def test_dr_wrapper_buffer_bob(env: SimEnv, selection: str):
    param_init = env.domain_param
    randomizer = create_default_randomizer(env)
    wrapper = DomainRandWrapperBuffer(env, randomizer, selection)
    # So far no randomization happened, thus the parameter should be equal
    assert env.domain_param == param_init
    assert wrapper._buffer is None

    # Randomize 5 times 5 new parameter sets
    for _ in range(5):
        wrapper.fill_buffer(10)
        for i in range(10):
            param_old = wrapper.domain_param
            if selection == "cyclic":
                assert wrapper._ring_idx == i
            else:
                assert 0 <= wrapper._ring_idx < len(wrapper.buffer)
            wrapper.reset()
            if selection == "cyclic":
                assert param_old != wrapper.domain_param
