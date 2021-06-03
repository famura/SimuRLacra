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

import pyrado


@pytest.mark.parametrize("base_seed", [-500, -1, 0b1_0000000000])
def test_out_of_bounds_base_seed(base_seed):
    with pytest.raises(
        pyrado.ValueErr, match=r"^base seed -?\d+ is not an unsigned 10-bit integer \(either too low or too high\)$"
    ):
        pyrado.set_seed(seed=base_seed, verbose=True)


@pytest.mark.parametrize(
    ["base_seed", "expected"],
    [
        (0b0000000000, 0b0000000000_00000000000000_00000000),
        (0b0101010101, 0b0101010101_00000000000000_00000000),
        (0b1111111111, 0b1111111111_00000000000000_00000000),
    ],
)
def test__base_seed(base_seed, expected):
    assert pyrado.set_seed(seed=base_seed, verbose=True) == expected


@pytest.mark.parametrize("multiplier", [-1, +1])
@pytest.mark.parametrize(
    ["sub_seed", "expected"],
    [
        (0b00000000000000, 0b0000000000_00000000000000_00000000),
        (0b01010101010101, 0b0000000000_01010101010101_00000000),
        (0b11111111111111, 0b0000000000_11111111111111_00000000),
        (0b1_00000000000011, 0b0000000000_00000000000011_00000000),
    ],
)
def test_sub_seed(multiplier, sub_seed, expected):
    assert pyrado.set_seed(seed=0, sub_seed=multiplier * sub_seed, verbose=True) == expected


@pytest.mark.parametrize("multiplier", [-1, +1])
@pytest.mark.parametrize(
    ["sub_sub_seed", "expected"],
    [
        (0b00000000, 0b0000000000_00000000000000_00000000),
        (0b01010101, 0b0000000000_00000000000000_01010101),
        (0b11111111, 0b0000000000_00000000000000_11111111),
        (0b1_00000110, 0b0000000000_00000000000001_00000110),
        (0b1_0000000000_00000000000001_00000110, 0b0000000000_00000000000001_00000110),
    ],
)
def test_sub_sub_seed(multiplier, sub_sub_seed, expected):
    assert pyrado.set_seed(seed=0, sub_sub_seed=multiplier * sub_sub_seed, verbose=True) == expected
