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
import hashlib

import pytest

import pyrado


@pytest.mark.parametrize(
    ["base_seed", "sub_seed", "sub_sub_seed", "expected"],
    [
        (0, None, None, 813134492),
        (0, None, 0, 813134492),
        (0, None, 1, 4276188331),
        (0, 0, None, 813134492),
        (0, 0, 0, 813134492),
        (0, 0, 1, 4276188331),
        (0, 1, None, 229607210),
        (0, 1, 0, 229607210),
        (0, 1, 1, 3918913762),
        (1, None, None, 532102107),
        (1, None, 0, 532102107),
        (1, None, 1, 2754337450),
        (1, 0, None, 532102107),
        (1, 0, 0, 532102107),
        (1, 0, 1, 2754337450),
        (1, 1, None, 713485941),
        (1, 1, 0, 713485941),
        (1, 1, 1, 3511146676),
    ],
)
def test_out_of_bounds_base_seed(base_seed, sub_seed, sub_sub_seed, expected):
    assert pyrado.set_seed(base_seed, sub_seed, sub_sub_seed, verbose=True) == expected
    assert pyrado.get_base_seed() == base_seed
