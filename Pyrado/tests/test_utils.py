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
import torch as to
import torch.nn as nn
from functools import partial
from math import ceil
from tqdm import tqdm

from pyrado.sampling.utils import gen_batch_idcs, gen_ordered_batch_idcs, gen_ordered_batches
from pyrado.utils.data_types import *
from pyrado.utils.functions import noisy_nonlin_fcn
from pyrado.utils.math import cosine_similarity, cov
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.policies.dummy import DummyPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.nn_layers import IndiNonlinLayer
from pyrado.utils.optimizers import GSS
from pyrado.utils.averaging import RunningExpDecayingAverage, RunningMemoryAverage
from pyrado.utils.standardizing import RunningStandardizer, Standardizer
from pyrado.utils.normalizing import RunningNormalizer, normalize


@pytest.mark.parametrize(
    'x, data_along_rows', [
        (np.random.rand(100, 4), True),
        (np.random.rand(4, 100), False)
    ], ids=['100_4', '4_100']
)
def test_cov(x, data_along_rows):
    rowvar = not data_along_rows
    cov_np = np.cov(x, rowvar=rowvar)
    cov_pyrado = cov(to.from_numpy(x), data_along_rows=data_along_rows).numpy()

    assert cov_pyrado.shape[0] == cov_pyrado.shape[1]
    if data_along_rows:
        assert cov_np.shape[0] == x.shape[1]
        assert cov_pyrado.shape[0] == x.shape[1]
    else:
        assert cov_np.shape[0] == x.shape[0]
        assert cov_pyrado.shape[0] == x.shape[0]
    assert np.allclose(cov_np, cov_pyrado)


@pytest.mark.parametrize(
    'env, expl_strat', [
        (BallOnBeamSim(dt=0.02, max_steps=100),
         DummyPolicy(BallOnBeamSim(dt=0.02, max_steps=100).spec)),
    ], ids=['bob_dummy']
)
def test_concat_rollouts(env, expl_strat):
    ro1 = rollout(env, expl_strat)
    ro2 = rollout(env, expl_strat)
    ro_cat = StepSequence.concat([ro1, ro2])
    assert isinstance(ro_cat, StepSequence)
    assert ro_cat.length == ro1.length + ro2.length


@pytest.mark.parametrize(
    'x, y', [
        (to.tensor([1., 2., 3.]), to.tensor([1., 2., 3.])),
        (to.tensor([1., 0., 1.]), to.tensor([1., 1e12, 1.])),
        (to.tensor([0., 0., 0.]), to.tensor([1., 2, 3.])),
        (to.tensor([1., 2., 3.]), to.tensor([2., 4., 6.])),
        (to.tensor([1., 2., 3.]), to.tensor([-1., -2., -3.])),
    ], ids=['same', 'similarity_1', 'similarity_0', 'colinear_scaled', 'colinear_opposite']
)
def test_cosine_similarity(x, y):
    # Only tested for vector inputs
    d_cos = cosine_similarity(x, y)
    assert isinstance(d_cos, to.Tensor)
    # The examples are chosen to result in 0, 1, or -1
    assert to.isclose(d_cos, to.tensor(0.)) or to.isclose(d_cos, to.tensor(1.)) or to.isclose(d_cos, to.tensor(-1.))


@pytest.mark.parametrize(
    'x, y', [
        ({'a': 1, 'b': 2}, {'c': 1, 'd': 4}),
        ({'a': 1, 'b': 2}, {'b': 3, 'd': 4}),
    ], ids=['disjoint', 'overlapping']
)
def test_merge_lod_var_dtype(x, y):
    z = merge_lod_var_dtype([x, y])
    assert z['a'] == 1
    if z['b'] == 2:  # disjoint
        assert z['c'] == 1
    elif z['b'] == 3:  # overlapping
        assert len(z) == 3
    else:
        assert False
    assert z['d'] == 4


@pytest.mark.parametrize(
    'batch_size, data_size', [
        (3, 30),
        (3, 29),
        (3, 28),
        (2, 2)
    ], ids=['division_mod0', 'division_mod1', 'division_mod2', 'edge_case']
)
@pytest.mark.parametrize(
    'sorted', [True, False], ids=['sorted', 'unsorted']
)
def test_gen_batch_idcs(batch_size, data_size, sorted):
    generator = gen_batch_idcs(batch_size, data_size)
    unordered_batches = list(generator)
    assert len(unordered_batches) == ceil(data_size/batch_size)
    assert all(len(uob) <= batch_size for uob in unordered_batches)

    generator = gen_ordered_batch_idcs(batch_size, data_size, sorted)
    ordered_batches = list(generator)
    assert len(ordered_batches) == ceil(data_size/batch_size)
    assert all(len(ob) <= batch_size for ob in ordered_batches)
    # Check if each mini-batch is sorted
    assert all(all(ob[i] <= ob[i + 1] for i in range(len(ob) - 1)) for ob in ordered_batches)


@pytest.mark.parametrize('dtype', ['torch', 'numpy'], ids=['to', 'np'])
@pytest.mark.parametrize('axis', [0, 1], ids=['ax_0', 'ax_1'])
def test_normalize(dtype, axis):
    for _ in range(10):
        x = to.rand(5, 3) if dtype == 'torch' else np.random.rand(5, 3)
        x_norm = normalize(x, axis=axis, order=1)
        if isinstance(x_norm, to.Tensor):
            x_norm = x_norm.numpy()  # for easier checking with pytest.approx
        assert np.sum(x_norm, axis=axis) == pytest.approx(1.)


@pytest.mark.parametrize(
    'data_seq, axis', [
        ([np.array([1, 1, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, -20, 20])], 0),
        ([np.array([1, 1, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, -20, 20])], None),
        ([np.array([1, 1, 2, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, 10, -20, 20])], 0),
        ([np.array([1, 1, 2, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, 10, -20, 20])], None),
        (
            [to.tensor([1., 1., 2]), to.tensor([1., 6., 3.]), to.tensor([1., 6., 3.]),
             to.tensor([10., -20., 20.])],
            0),
        (
            [to.tensor([1., 1., 2]), to.tensor([1., 6., 3.]), to.tensor([1., 6., 3.]),
             to.tensor([10., -20., 20.])],
            -1),
        (
            [to.tensor([1., 1, 2, 2]), to.tensor([1., 6, 3]), to.tensor([1., 6, 3]),
             to.tensor([10., 10, -20, 20])],
            0),
        (
            [to.tensor([1., 1, 2, 2]), to.tensor([1., 6, 3]), to.tensor([1., 6, 3]),
             to.tensor([10., 10, -20, 20])],
            -1),
    ], ids=['np_same_length_0', 'np_same_length_None', 'np_mixed_length_0', 'np_mixed_length_None',
            'to_same_length_0', 'to_same_length_-1', 'to_mixed_length_0', 'to_mixed_length_-1']
)
def test_running_standardizer(data_seq, axis):
    rs = RunningStandardizer()
    for data in data_seq:
        z = rs(data, axis)
        assert z is not None
    rs.reset()
    assert rs._mean is None and rs._sum_sq_diffs is None and rs._iter == 0


@pytest.mark.parametrize(
    'data_seq, alpha', [
        (
            [np.array([1, 1, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, -20, 20])],
            0.9
        ),
        (
            [to.tensor([1., 1., 2]), to.tensor([1., 6., 3.]), to.tensor([1., 6., 3.]), to.tensor([10., -20., 20.])],
            0.1
        ),
    ], ids=['np', 'to']
)
def test_running_expdecay_average(data_seq, alpha):
    reda = RunningExpDecayingAverage(alpha)
    for data in data_seq:
        z = reda(data)
        assert z is not None
    reda.reset(alpha=0.5)
    assert reda._alpha == 0.5 and reda._prev_est is None


@pytest.mark.parametrize(
    'data_seq, capacity', [
        (
            [np.array([1., 1, 2]), np.array([1., 1, 2]), np.array([1., 1, 2]), np.array([-2., -2, -4])],
            3
        ),
        (
            [to.tensor([1., 1, 2]), to.tensor([1., 1, 2]), to.tensor([1., 1, 2]), to.tensor([-2., -2, -4])],
            3
        ),
    ], ids=['np', 'to']
)
def test_running_mem_average(data_seq, capacity):
    rma = RunningMemoryAverage(capacity)
    for i, data in enumerate(data_seq):
        z = rma(data)
        if i <= 2:
            to.testing.assert_allclose(z, to.tensor([1., 1, 2]))  # works with PyTorch Tensors and numpy arrays
        elif i == 3:
            to.testing.assert_allclose(z, to.tensor([0., 0, 0]))  # works with PyTorch Tensors and numpy arrays
    rma.reset(capacity=5)
    assert rma.capacity == 5 and rma.memory is None


@pytest.mark.parametrize(
    'data_seq', [
        [5*np.random.rand(25, 3), 0.1*np.random.rand(5, 3), 20*np.random.rand(70, 3)],
        [5*to.rand(25, 3), 0.1*to.rand(5, 3), 20*to.rand(70, 3)]
    ], ids=['np', 'to']
)
def test_running_normalizer(data_seq):
    rn = RunningNormalizer()
    for data in data_seq:
        data_norm = rn(data)
        assert (-1 <= data_norm).all()
        assert (data_norm <= 1).all()


@pytest.mark.parametrize(
    'x', [
        to.rand(1000, 1),
        to.rand(1, 1000),
        to.rand(1000, 1000),
        np.random.rand(1, 1000),
        np.random.rand(1000, 1),
        np.random.rand(1000, 1000)
    ], ids=['to_1x1000', 'to_1000x1', 'to_1000x1000', 'np_1x1000', 'np_1000x1', 'np_1000x1000']
)
def test_stateful_standardizer(x):
    ss = Standardizer()

    if isinstance(x, to.Tensor):
        x_stdized = ss.standardize(x)
        assert x_stdized.shape == x.shape
        assert to.allclose(x_stdized.mean(), to.zeros(1))
        assert to.allclose(x_stdized.std(), to.ones(1))

        x_restrd = ss.unstandardize(x_stdized)
        assert x_restrd.shape == x.shape
        assert to.allclose(x_restrd, x, rtol=1e-02, atol=1e-05)

    elif isinstance(x, np.ndarray):
        x_stdized = ss.standardize(x)
        assert x_stdized.shape == x.shape
        assert np.allclose(x_stdized.mean(), np.zeros(1))
        assert np.allclose(x_stdized.std(), np.ones(1))

        x_restrd = ss.unstandardize(x_stdized)
        assert x_restrd.shape == x.shape
        assert np.allclose(x_restrd, x, rtol=1e-02, atol=1e-05)


@pytest.mark.parametrize(
    'g, ed', [
        (1., 2.),
        (np.array([-1., 2.]), np.eye(2))
    ], ids=['scalar', 'array']
)
def test_ds_spec(g, ed):
    # Base class
    dss = DSSpec(function='name', goal=g)
    assert isinstance(dss, dict)
    assert dss['function'] == 'name'
    if isinstance(g, np.ndarray):
        assert np.all(dss['goal'] == g)
    else:
        assert dss['goal'] == g

    # Linear first order subclass
    lds = LinDSSpec(function='lin', goal=g, errorDynamics=ed)
    assert isinstance(dss, dict)
    assert lds['function'] == 'lin'
    if isinstance(g, np.ndarray):
        assert np.all(lds['goal'] == g)
        assert np.all(lds['errorDynamics'] == ed)
    else:
        assert lds['goal'] == g
        assert lds['errorDynamics'] == ed

    # Mass-Spring-Damper subclass
    msds = MSDDSSpec(function='msd', goal=g, damping=2., attractorStiffness=3., mass=4.)
    assert isinstance(dss, dict)
    assert msds['function'] == 'msd'
    if isinstance(g, np.ndarray):
        assert np.all(msds['goal'] == g)
    else:
        assert msds['goal'] == g
    assert msds['damping'] == 2.
    assert msds['attractorStiffness'] == 3.
    assert msds['mass'] == 4.


@pytest.mark.visualization
@pytest.mark.parametrize(
    'identical_bounds', [
        True, False
    ], ids=['identical', 'separate']
)
def test_gss_optimizer_identical_bounds(identical_bounds):
    class Dummy:
        def loss_fcn(self):
            # Some function to minimize
            return (self.x + self.y + 4)**2

        def __init__(self):
            # Test with different lower and upper bounds
            self.x, self.y = to.tensor([0.]), to.tensor([4.])
            x_min, x_max = to.tensor([-10.]), to.tensor([5.])
            if identical_bounds:
                self.optim = GSS([{'params': self.x}, {'params': self.y}], x_min, x_max)
            else:
                x_min_override = to.tensor([-6.])
                self.optim = GSS([{'params': self.x, 'param_min': x_min_override}, {'params': self.y}], x_min, x_max)
            print(self.optim)

    dummy = Dummy()

    for i in range(2):
        dummy.optim.step(dummy.loss_fcn)
    assert dummy.x != dummy.y
    print(f'x = {dummy.x.item()} \t y = {dummy.y.item()}')


def test_gss_optimizer_functional():
    class Dummy:
        def loss_fcn(self):
            # Some function to minimize
            return (self.x + 4)**2

        def __init__(self):
            # Test with different lower and upper bounds
            self.x = to.tensor([0.])
            x_min, x_max = to.tensor([-10.]), to.tensor([10.])
            self.optim = GSS([{'params': self.x}], x_min, x_max)

    dummy = Dummy()

    for i in range(100):
        dummy.optim.step(dummy.loss_fcn)
    assert to.norm(dummy.x + 4) < 1e-4


@pytest.mark.visualization
def test_gss_optimizer_nlin_fcn():
    from matplotlib import pyplot as plt
    # Parameters
    x_grid = to.linspace(-2., 3., 200)
    f = 1.
    noise_std = 0.1

    # Init param and optimizer
    x_init = to.rand(1)*(x_grid.max() - x_grid.min())/2 + x_grid.min() + (x_grid.max() - x_grid.min())/4  # [.25, .75]
    x = nn.Parameter(to.tensor([x_init]), requires_grad=False)
    optim = GSS([x], param_min=x_grid.min().unsqueeze(0), param_max=x_grid.max().unsqueeze(0))
    obj_fcn = partial(noisy_nonlin_fcn, x=x, f=f, noise_std=noise_std)
    num_epochs = 10

    # Init plotting
    fig = plt.figure()
    plt.plot(x_grid, noisy_nonlin_fcn(x=x_grid, f=f), label='noise free fcn')
    plt.scatter(x.data.numpy(), obj_fcn().numpy(), s=40, marker='x', color='k', label='init guess')
    colors = plt.get_cmap('inferno')(np.linspace(0, 1, num_epochs))

    for e in tqdm(range(num_epochs), total=num_epochs):
        # Evaluate at a the current point
        optim.step(obj_fcn)

        # Plot current evaluation
        plt.plot(x_grid, noisy_nonlin_fcn(x=x_grid, f=f, noise_std=noise_std), alpha=0.2)
        plt.scatter(x.data.numpy(), obj_fcn().numpy(), s=16, color=colors[e])

    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend()
    plt.show()
    assert noisy_nonlin_fcn(x, f=f, noise_std=noise_std) < noisy_nonlin_fcn(x_init, f=f, noise_std=noise_std)


@pytest.mark.parametrize('in_features', [1, 3], ids=['1dim', '3dim'])
@pytest.mark.parametrize('same_nonlin', [True, False], ids=['same_nonlin', 'different_nonlin'])
@pytest.mark.parametrize('bias', [True, False], ids=['bias', 'no_bias'])
@pytest.mark.parametrize('weight', [True, False], ids=['weight', 'no_weight'])
def test_indi_nonlin_layer(in_features, same_nonlin, bias, weight):
    if not same_nonlin and in_features > 1:
        nonlin = in_features*[to.tanh]
    else:
        nonlin = to.sigmoid
    layer = IndiNonlinLayer(in_features, nonlin, bias, weight)
    assert isinstance(layer, nn.Module)

    i = to.randn(in_features)
    o = layer(i)
    assert isinstance(o, to.Tensor)
    assert i.shape == o.shape
