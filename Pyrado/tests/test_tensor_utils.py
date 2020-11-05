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
import torch as to

from pyrado.utils.tensor import stack_tensor_list, stack_tensor_dict_list, insert_tensor_col, \
    atleast_2D, atleast_3D


@pytest.mark.parametrize(
    'x', [
        to.tensor(3.),
        to.rand(1, ),
        to.rand(2, ),
        to.rand(1, 2),
        to.rand(2, 1),
        to.rand(2, 3),
        to.rand(2, 3, 4)
    ],
    ids=['sclar', 'scalar_1D', 'vec_1D', 'vec_2D', 'vec_2D_T', 'arr_2D', 'arr_3D'])
def test_atleast_2D(x):
    x_al2d = atleast_2D(x)
    assert x_al2d.ndim >= 2

    # We want to mimic the numpy function
    x_np = np.atleast_2d(x.numpy())
    assert np.all(x_al2d.numpy() == x_np)


@pytest.mark.parametrize(
    'x', [
        to.tensor(3.),
        to.rand(1, ),
        to.rand(2, ),
        to.rand(1, 2),
        to.rand(2, 1),
        to.rand(2, 3),
        to.rand(2, 3, 4)
    ],
    ids=['sclar', 'scalar_1D', 'vec_1D', 'vec_2D', 'vec_2D_T', 'arr_2D', 'arr_3D'])
def test_atleast_3D(x):
    x_al3d = atleast_3D(x)
    assert x_al3d.ndim >= 3

    # We want to mimic the numpy function
    x_np = np.atleast_3d(x.numpy())
    assert np.all(x_al3d.numpy() == x_np)


def test_stack_tensors():
    tensors = [
        to.tensor([1, 2, 3]),
        to.tensor([2, 3, 4]),
        to.tensor([4, 5, 6]),
    ]

    stack = stack_tensor_list(tensors)

    to.testing.assert_allclose(stack, to.tensor([
        [1, 2, 3],
        [2, 3, 4],
        [4, 5, 6],
    ]))


def test_stack_tensors_scalar():
    tensors = [1, 2, 3]
    stack = stack_tensor_list(tensors)
    to.testing.assert_allclose(stack, to.tensor([1, 2, 3]))


def test_stack_tensor_dicts():
    tensors = [
        {'multi': [1, 2], 'single': 1},
        {'multi': [3, 4], 'single': 2},
        {'multi': [5, 6], 'single': 3},
    ]
    stack = stack_tensor_dict_list(tensors)
    to.testing.assert_allclose(stack['single'], to.tensor([1, 2, 3]))
    to.testing.assert_allclose(stack['multi'], to.tensor([[1, 2], [3, 4], [5, 6]]))


@pytest.mark.parametrize(
    'orig, col', [
        (to.rand((1, 1)), to.zeros(1, 1)),
        (to.rand((3, 3)), to.zeros(3, 1)),
    ],
    ids=['1_dim', '3_dim']
)
def test_insert_tensor_col(orig, col):
    for col_idx in range(orig.shape[1] + 1):  # also check appending case
        result = insert_tensor_col(orig, col_idx, col)
        # Check number of rows and columns
        assert orig.shape[0] == result.shape[0]
        assert orig.shape[1] == result.shape[1] - 1
        # Check the values
        to.testing.assert_allclose(result[:, col_idx], col.squeeze())
