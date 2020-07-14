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

from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, DomainRandWrapperBuffer


@pytest.mark.wrapper
def test_dr_wrapper_live_bob(default_bob, bob_pert):
    param_init = default_bob.domain_param
    wrapper = DomainRandWrapperLive(default_bob, bob_pert)
    # So far no randomization happened, thus the parameter should be equal
    assert default_bob.domain_param == param_init

    # Randomize 10 times 1 new parameter set
    for _ in range(10):
        param_old = wrapper.domain_param
        wrapper.reset()
        assert param_old != wrapper.domain_param


@pytest.mark.wrapper
def test_dr_wrapper_buffer_bob(default_bob, bob_pert):
    param_init = default_bob.domain_param
    wrapper = DomainRandWrapperBuffer(default_bob, bob_pert)
    # So far no randomization happened, thus the parameter should be equal
    assert default_bob.domain_param == param_init
    assert wrapper._buffer is None

    # Randomize 10 times 13 new parameter sets
    for _ in range(10):
        wrapper.fill_buffer(13)
        for i in range(13):
            param_old = wrapper.domain_param
            assert wrapper._ring_idx == i
            wrapper.reset()
            assert param_old != wrapper.domain_param
