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

import math
import numpy as np
import pytest

from pyrado.spaces.box import BoxSpace
from pyrado.spaces.discrete import DiscreteSpace
from pyrado.spaces.polar import Polar2DPosVelSpace


@pytest.fixture(
    scope="function",
    params=[
        (-np.ones((7,)), np.ones((7,))),
        (-np.ones((7, 1)), np.ones((7, 1))),
        (np.array([5, -math.pi / 2, -math.pi]), np.array([5, math.pi / 2, math.pi])),
    ],
    ids=["box_flatdim", "box", "half_sphere"],
)
def bs(request):
    return BoxSpace(request.param[0], request.param[1])


@pytest.fixture(
    scope="function",
    params=[
        np.array([1]),
        np.array([[1]]),
        np.array([1, 2, 3], dtype=np.int32),
        np.array([-2, -1, 0, 1, 2], dtype=np.int64),
        np.array([4, -3, 5, 0, 1, 2, 6, -7], dtype=np.int32),
        np.array([4.0, -3, 5, 0, 1, 2, 6, -7], dtype=np.float),
    ],
    ids=["scalar1dim", "scalar2dim", "pos", "pos_neg", "prandom", "prandom_float"],
)
def ds(request):
    return DiscreteSpace(request.param)


def test_sample_contains_box_space(bs):
    for _ in range(10):
        ele = bs.sample_uniform()
        assert bs.contains(ele)


def test_contains_verbose_box_space():
    bs = BoxSpace([-1, -2, -3], [1, 2, 3])
    not_ele = np.array([-4, 0, 4])
    f = bs.contains(not_ele, verbose=True)
    assert not f


def test_copy_box_space(bs):
    bsc = bs.copy()
    assert id(bs) != id(bsc)
    bs.bound_lo *= -3
    assert np.all(bs.bound_lo != bsc.bound_lo)
    bsc.bound_up *= 5
    assert np.all(bs.bound_up != bsc.bound_up)


def test_project_to_box_space(bs):
    for _ in range(100):
        # Sample from within the space w.p. 1/5 and from outside the space w.p. 4/5
        ele_s = bs.sample_uniform() * 5.0
        ele_p = bs.project_to(ele_s)
        assert bs.contains(ele_p)


def test_flat_dim_box_space(bs):
    print(bs.flat_dim)


@pytest.mark.parametrize(
    "idcs",
    [
        [0, 1, 2],
        [0, 2],
    ],
    ids=["3_wo_gap", "2_w_gap"],
)
def test_subspace_box_space(bs, idcs):
    subbox = bs.subspace(idcs)
    if len(bs.shape) == 1:
        assert subbox.flat_dim == len(idcs)
        np.testing.assert_equal(subbox.bound_lo, bs.bound_lo[idcs])
        np.testing.assert_equal(subbox.bound_up, bs.bound_up[idcs])
        np.testing.assert_equal(subbox.labels, bs.labels[idcs])
    elif len(bs.shape) == 2:
        assert subbox.flat_dim == len(idcs) * bs.shape[1]
        np.testing.assert_equal(subbox.bound_lo, bs.bound_lo[idcs, :])
        np.testing.assert_equal(subbox.bound_up, bs.bound_up[idcs, :])
        np.testing.assert_equal(subbox.labels, bs.labels[idcs, :])


@pytest.mark.parametrize(
    "bs_list",
    [
        [BoxSpace([-1, -2, -3], [1, 2, 3]), BoxSpace([-11, -22, -33], [11, 22, 33])],
        [BoxSpace([-1], [1]), BoxSpace([-22, 33], [22, 33])],
    ],
    ids=["identical_sizes", "different_sizes"],
)
def test_cat_box_space(bs_list):
    bs_cat = BoxSpace.cat(bs_list)
    assert isinstance(bs_cat, BoxSpace)
    assert bs_cat.flat_dim == sum([bs.flat_dim for bs in bs_list])


@pytest.mark.parametrize(
    "ds_list",
    [
        [DiscreteSpace([-1, -2, -3]), DiscreteSpace([11, 22, 33])],
        [DiscreteSpace([-1]), DiscreteSpace([22, 33])],
    ],
    ids=["identical_sizes", "different_sizes"],
)
def test_cat_discrete_space(ds_list):
    ds_cat = DiscreteSpace.cat(ds_list)
    assert isinstance(ds_cat, DiscreteSpace)
    assert ds_cat.num_ele == sum([ds.num_ele for ds in ds_list])


def test_sample_contains_discrete_space(ds):
    for _ in range(10):
        ele = ds.sample_uniform()
        assert ds.contains(ele)


def test_copy_discrete_space(ds):
    dsc = ds.copy()
    assert id(ds) != id(dsc)
    ds.bound_lo *= -3
    assert np.all(ds.bound_lo != dsc.bound_lo)
    dsc.bound_up *= 5
    assert np.all(ds.bound_up != dsc.bound_up)


def test_project_to_discrete_space(ds):
    for _ in range(100):
        # Sample from within the space and from outside the space
        ele_s = ds.sample_uniform() * 5.0
        ele_p = ds.project_to(ele_s)
        assert ds.contains(ele_p)


def test_torus2D():
    # Parametrize within the test to hape tailored asserts
    torus_0deg = Polar2DPosVelSpace(np.array([1, 0, -0.1, -0.1]), np.array([1, 0, 0.1, 0.1]))
    sample_0deg = torus_0deg.sample_uniform()
    assert sample_0deg[0] == 1  # x position
    assert sample_0deg[1] == 0  # y position

    torus_90deg = Polar2DPosVelSpace(np.array([1, np.pi / 2, 0, 0]), np.array([1, np.pi / 2, 0, 0]))
    sample_90deg = torus_90deg.sample_uniform()
    assert np.all(np.isclose(sample_90deg, np.array([0, 1, 0, 0])))
