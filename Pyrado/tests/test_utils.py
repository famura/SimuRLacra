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

import os.path as osp
from functools import partial
from math import ceil

import pytest
import torch.nn as nn
from tqdm import tqdm

from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.iteration import IterationTracker
from pyrado.policies.feed_forward.dummy import DummyPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.utils import gen_ordered_batch_idcs, gen_ordered_batches, gen_shuffled_batch_idcs
from pyrado.spaces import BoxSpace
from pyrado.utils.averaging import RunningExpDecayingAverage, RunningMemoryAverage
from pyrado.utils.checks import check_all_lengths_equal, check_all_shapes_equal, check_all_types_equal, is_iterator
from pyrado.utils.data_processing import (
    MinMaxScaler,
    RunningNormalizer,
    RunningStandardizer,
    Standardizer,
    correct_atleast_2d,
    normalize,
    scale_min_max,
)
from pyrado.utils.data_types import *
from pyrado.utils.functions import noisy_nonlin_fcn, skyline
from pyrado.utils.input_output import completion_context, print_cbt_once
from pyrado.utils.math import cosine_similarity, cov, logmeanexp, numerical_differentiation_coeffs, rmse
from pyrado.utils.optimizers import GSS


@pytest.mark.parametrize(
    "x, data_along_rows", [(np.random.rand(100, 4), True), (np.random.rand(4, 100), False)], ids=["100_4", "4_100"]
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
    "env, expl_strat",
    [
        (BallOnBeamSim(dt=0.02, max_steps=100), DummyPolicy(BallOnBeamSim(dt=0.02, max_steps=100).spec)),
    ],
    ids=["bob_dummy"],
)
def test_concat_rollouts(env, expl_strat):
    ro1 = rollout(env, expl_strat)
    ro2 = rollout(env, expl_strat)
    ro_cat = StepSequence.concat([ro1, ro2])
    assert isinstance(ro_cat, StepSequence)
    assert ro_cat.length == ro1.length + ro2.length


@pytest.mark.parametrize(
    "x, y",
    [
        (to.tensor([1.0, 2.0, 3.0]), to.tensor([1.0, 2.0, 3.0])),
        (to.tensor([1.0, 0.0, 1.0]), to.tensor([1.0, 1e12, 1.0])),
        (to.tensor([0.0, 0.0, 0.0]), to.tensor([1.0, 2, 3.0])),
        (to.tensor([1.0, 2.0, 3.0]), to.tensor([2.0, 4.0, 6.0])),
        (to.tensor([1.0, 2.0, 3.0]), to.tensor([-1.0, -2.0, -3.0])),
    ],
    ids=["same", "similarity_1", "similarity_0", "colinear_scaled", "colinear_opposite"],
)
def test_cosine_similarity(x, y):
    # Only tested for vector inputs
    d_cos = cosine_similarity(x, y)
    assert isinstance(d_cos, to.Tensor)
    # The examples are chosen to result in 0, 1, or -1
    assert to.isclose(d_cos, to.tensor(0.0)) or to.isclose(d_cos, to.tensor(1.0)) or to.isclose(d_cos, to.tensor(-1.0))


@pytest.mark.parametrize("type", ["numpy", "torch"], ids=["numpy", "torch"])
@pytest.mark.parametrize("dim", [0, 1], ids=["dim0", "dim1"])
def test_rmse(type, dim):
    shape = (42, 21)
    if type == "numpy":
        x = np.random.randn(*shape)
        y = np.random.randn(*shape)
    else:
        x = to.randn(*shape)
        y = to.randn(*shape)
    e = rmse(x, y)
    if type == "numpy":
        assert isinstance(e, np.ndarray)
    else:
        assert isinstance(e, to.Tensor)


@pytest.mark.parametrize(
    "x, y",
    [
        ({"a": 1, "b": 2}, {"c": 1, "d": 4}),
        ({"a": 1, "b": 2}, {"b": 3, "d": 4}),
    ],
    ids=["disjoint", "overlapping"],
)
def test_merge_lod_var_dtype(x, y):
    z = merge_dicts([x, y])
    assert z["a"] == 1
    if z["b"] == 2:  # disjoint
        assert z["c"] == 1
    elif z["b"] == 3:  # overlapping
        assert len(z) == 3
    else:
        assert False
    assert z["d"] == 4


@pytest.mark.parametrize(
    "seq",
    [
        [1, 2, 3],
        (1, 2, 3),
    ],
    ids=["list", "tuple"],
)
@pytest.mark.parametrize("num_reps", [3])
def test_repeat_interleave(seq, num_reps):
    seq_rep = repeat_interleave(seq, num_reps)
    assert len(seq_rep) == num_reps * len(seq)
    assert all([seq_rep[i] == seq[0] for i in range(num_reps)])  # only check first ele
    assert type(seq_rep) == type(seq)


@pytest.mark.parametrize(
    "batch_size, data_size",
    [(3, 30), (3, 29), (3, 28), (2, 2)],
    ids=["division_mod0", "division_mod1", "division_mod2", "edge_case"],
)
@pytest.mark.parametrize("sorted", [True, False], ids=["sorted", "unsorted"])
def test_gen_batch_idcs(batch_size, data_size, sorted):
    generator = gen_shuffled_batch_idcs(batch_size, data_size)
    unordered_batches = list(generator)
    assert len(unordered_batches) == ceil(data_size / batch_size)
    assert all(len(uob) <= batch_size for uob in unordered_batches)

    generator = gen_ordered_batch_idcs(batch_size, data_size, sorted)
    ordered_batches = list(generator)
    assert len(ordered_batches) == ceil(data_size / batch_size)
    assert all(len(ob) <= batch_size for ob in ordered_batches)
    # Check if each mini-batch is sorted
    assert all(all(ob[i] <= ob[i + 1] for i in range(len(ob) - 1)) for ob in ordered_batches)


@pytest.mark.parametrize(
    "data, batch_size",
    [
        (list(range(9)), 3),
        (list(range(10)), 3),
    ],
    ids=["division_mod0", "division_mod1"],
)
def test_gen_ordered_batches(data, batch_size):
    n = ceil(len(data) / batch_size)
    for i, batch in enumerate(gen_ordered_batches(data, batch_size)):
        if i < n - 1:
            assert len(batch) == batch_size
        if i == n - 1:
            assert len(batch) <= batch_size


@pytest.mark.parametrize("dtype", ["torch", "numpy"], ids=["to", "np"])
@pytest.mark.parametrize("axis", [0, 1], ids=["ax_0", "ax_1"])
def test_normalize(dtype, axis):
    for _ in range(10):
        x = to.rand(5, 3) if dtype == "torch" else np.random.rand(5, 3)
        x_norm = normalize(x, axis=axis, order=1)
        if isinstance(x_norm, to.Tensor):
            x_norm = x_norm.numpy()  # for easier checking with pytest.approx
        assert np.sum(x_norm, axis=axis) == pytest.approx(1.0)


@pytest.mark.parametrize("dtype", ["torch", "numpy", "mixed"], ids=["to", "np", "mixed"])
@pytest.mark.parametrize(
    "lb, ub",
    [(0, 1), (-1, 1), (-2.5, 0), (-np.ones((3, 2)), np.ones((3, 2)))],
    ids=["lb0_ub1", "lb-1_ub1", "lb-2.5_ub0", "np_ones"],
)
def test_scale_min_max(dtype, lb, ub):
    for _ in range(10):
        if dtype == "torch":
            bound_lo = to.tensor([lb], dtype=to.float64)
            bound_up = to.tensor([ub], dtype=to.float64)
        elif dtype == "numpy":
            bound_lo = np.array(lb, dtype=np.float64)
            bound_up = np.array(ub, dtype=np.float64)
        else:
            bound_lo = lb
            bound_up = ub

        x = 1e2 * to.rand(3, 2) if dtype == "torch" else np.random.rand(3, 2)
        x_scaled = scale_min_max(x, bound_lo, bound_up)
        if isinstance(x_scaled, to.Tensor) and isinstance(bound_lo, to.Tensor) and isinstance(bound_up, to.Tensor):
            x_scaled = x_scaled.numpy()  # for easier checking with pytest.approx
            bound_lo = bound_lo.numpy()
            bound_up = bound_up.numpy()
        assert np.all(bound_lo * np.ones_like(x_scaled) <= x_scaled)
        assert np.all(x_scaled <= bound_up * np.ones_like(x_scaled))


@pytest.mark.parametrize("dtype", ["torch", "numpy"], ids=["to", "np"])
@pytest.mark.parametrize(
    "lb, ub",
    [(0, 1), (-1, 1), (-2.5, 0), (-np.ones((3, 2)), np.ones((3, 2)))],
    ids=["lb0_ub1", "lb-1_ub1", "lb-2.5_ub0", "np_ones"],
)
def test_minmaxscaler(dtype, lb, ub):
    for _ in range(10):
        scaler = MinMaxScaler(lb, ub)
        x = 1e2 * to.rand(3, 2) if dtype == "torch" else np.random.rand(3, 2)
        x_scaled = scaler.scale_to(x)
        x_scaled_back = scaler.scale_back(x_scaled)

        if isinstance(x_scaled, to.Tensor):
            x_scaled = x_scaled.numpy()  # for easier checking with pytest.approx
            x_scaled_back = x_scaled_back.numpy()  # for easier checking with pytest.approx
        assert np.all(scaler._bound_lo * np.ones_like(x_scaled) <= x_scaled)
        assert np.all(x_scaled <= scaler._bound_up * np.ones_like(x_scaled))
        assert np.allclose(x, x_scaled_back)


@pytest.mark.parametrize(
    "data_seq, axis",
    [
        ([np.array([1, 1, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, -20, 20])], 0),
        ([np.array([1, 1, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, -20, 20])], None),
        ([np.array([1, 1, 2, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, 10, -20, 20])], 0),
        ([np.array([1, 1, 2, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, 10, -20, 20])], None),
        (
            [
                to.tensor([1.0, 1.0, 2]),
                to.tensor([1.0, 6.0, 3.0]),
                to.tensor([1.0, 6.0, 3.0]),
                to.tensor([10.0, -20.0, 20.0]),
            ],
            0,
        ),
        (
            [
                to.tensor([1.0, 1.0, 2]),
                to.tensor([1.0, 6.0, 3.0]),
                to.tensor([1.0, 6.0, 3.0]),
                to.tensor([10.0, -20.0, 20.0]),
            ],
            -1,
        ),
        (
            [to.tensor([1.0, 1, 2, 2]), to.tensor([1.0, 6, 3]), to.tensor([1.0, 6, 3]), to.tensor([10.0, 10, -20, 20])],
            0,
        ),
        (
            [to.tensor([1.0, 1, 2, 2]), to.tensor([1.0, 6, 3]), to.tensor([1.0, 6, 3]), to.tensor([10.0, 10, -20, 20])],
            -1,
        ),
    ],
    ids=[
        "np_same_length_0",
        "np_same_length_None",
        "np_mixed_length_0",
        "np_mixed_length_None",
        "to_same_length_0",
        "to_same_length_-1",
        "to_mixed_length_0",
        "to_mixed_length_-1",
    ],
)
def test_running_standardizer(data_seq, axis):
    rs = RunningStandardizer()
    for data in data_seq:
        z = rs(data, axis)
        assert z is not None
    rs.reset()
    assert rs.mean is None and rs.sum_sq_diffs is None and rs.iter == 0


@pytest.mark.parametrize(
    "data_seq, alpha",
    [
        ([np.array([1, 1, 2]), np.array([1, 6, 3]), np.array([1, 6, 3]), np.array([10, -20, 20])], 0.9),
        (
            [
                to.tensor([1.0, 1.0, 2]),
                to.tensor([1.0, 6.0, 3.0]),
                to.tensor([1.0, 6.0, 3.0]),
                to.tensor([10.0, -20.0, 20.0]),
            ],
            0.1,
        ),
    ],
    ids=["np", "to"],
)
def test_running_expdecay_average(data_seq, alpha):
    reda = RunningExpDecayingAverage(alpha)
    for data in data_seq:
        z = reda(data)
        assert z is not None
    reda.reset(alpha=0.5)
    assert reda._alpha == 0.5 and reda._prev_est is None


@pytest.mark.parametrize(
    "data_seq, capacity",
    [
        ([np.array([1.0, 1, 2]), np.array([1.0, 1, 2]), np.array([1.0, 1, 2]), np.array([-2.0, -2, -4])], 3),
        ([to.tensor([1.0, 1, 2]), to.tensor([1.0, 1, 2]), to.tensor([1.0, 1, 2]), to.tensor([-2.0, -2, -4])], 3),
    ],
    ids=["np", "to"],
)
def test_running_mem_average(data_seq, capacity):
    rma = RunningMemoryAverage(capacity)
    for i, data in enumerate(data_seq):
        z = rma(data)
        if isinstance(z, np.ndarray):
            z = z.astype(dtype=np.float32)
        if i <= 2:
            assert z == pytest.approx([1.0, 1, 2])
        elif i == 3:
            assert z == pytest.approx([0.0, 0, 0])
    rma.reset(capacity=5)
    assert rma.capacity == 5 and rma.memory is None


@pytest.mark.parametrize(
    "data_seq",
    [
        [5 * np.random.rand(25, 3), 0.1 * np.random.rand(5, 3), 20 * np.random.rand(70, 3)],
        [5 * to.rand(25, 3), 0.1 * to.rand(5, 3), 20 * to.rand(70, 3)],
    ],
    ids=["np", "to"],
)
def test_running_normalizer(data_seq):
    rn = RunningNormalizer()
    for data in data_seq:
        data_norm = rn(data)
        assert (-1 <= data_norm).all()
        assert (data_norm <= 1).all()


@pytest.mark.parametrize(
    "x",
    [
        to.rand(1000, 1),
        to.rand(1, 1000),
        to.rand(1000, 1000),
        np.random.rand(1, 1000),
        np.random.rand(1000, 1),
        np.random.rand(1000, 1000),
    ],
    ids=["to_1x1000", "to_1000x1", "to_1000x1000", "np_1x1000", "np_1000x1", "np_1000x1000"],
)
def test_stateful_standardizer(x):
    ss = Standardizer()

    if isinstance(x, to.Tensor):
        x_stdized = ss.standardize(x)
        assert x_stdized.shape == x.shape
        assert to.allclose(x_stdized.mean(), to.zeros(1), atol=1e-6)
        assert to.allclose(x_stdized.std(), to.ones(1), atol=1e-6)

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


@pytest.mark.parametrize("g, ed", [(1.0, 2.0), (np.array([-1.0, 2.0]), np.eye(2))], ids=["scalar", "array"])
def test_ds_spec(g, ed):
    # Base class
    dss = DSSpec(function="name", goal=g)
    assert isinstance(dss, dict)
    assert dss["function"] == "name"
    if isinstance(g, np.ndarray):
        assert np.all(dss["goal"] == g)
    else:
        assert dss["goal"] == g

    # Linear first order subclass
    lds = LinDSSpec(function="lin", goal=g, errorDynamics=ed)
    assert isinstance(dss, dict)
    assert lds["function"] == "lin"
    if isinstance(g, np.ndarray):
        assert np.all(lds["goal"] == g)
        assert np.all(lds["errorDynamics"] == ed)
    else:
        assert lds["goal"] == g
        assert lds["errorDynamics"] == ed

    # Mass-Spring-Damper subclass
    msds = MSDDSSpec(function="msd", goal=g, damping=2.0, attractorStiffness=3.0, mass=4.0)
    assert isinstance(dss, dict)
    assert msds["function"] == "msd"
    if isinstance(g, np.ndarray):
        assert np.all(msds["goal"] == g)
    else:
        assert msds["goal"] == g
    assert msds["damping"] == 2.0
    assert msds["attractorStiffness"] == 3.0
    assert msds["mass"] == 4.0


@pytest.mark.parametrize("identical_bounds", [True, False], ids=["identical", "separate"])
def test_gss_optimizer_identical_bounds(identical_bounds):
    class Dummy:
        def loss_fcn(self):
            # Some function to minimize
            return (self.x + self.y + 4) ** 2

        def __init__(self):
            # Test with different lower and upper bounds
            self.x, self.y = to.tensor([0.0]), to.tensor([4.0])
            x_min, x_max = to.tensor([-10.0]), to.tensor([5.0])
            if identical_bounds:
                self.optim = GSS([{"params": self.x}, {"params": self.y}], x_min, x_max)
            else:
                x_min_override = to.tensor([-6.0])
                self.optim = GSS([{"params": self.x, "param_min": x_min_override}, {"params": self.y}], x_min, x_max)

    dummy = Dummy()

    for i in range(2):
        dummy.optim.step(dummy.loss_fcn)
    assert dummy.x != dummy.y
    print(f"x = {dummy.x.item()} \t y = {dummy.y.item()}")


def test_gss_optimizer_functional():
    class Dummy:
        def loss_fcn(self):
            # Some function to minimize
            return (self.x + 4) ** 2

        def __init__(self):
            # Test with different lower and upper bounds
            self.x = to.tensor([0.0])
            x_min, x_max = to.tensor([-10.0]), to.tensor([10.0])
            self.optim = GSS([{"params": self.x}], x_min, x_max)

    dummy = Dummy()

    for i in range(100):
        dummy.optim.step(dummy.loss_fcn)
    assert to.norm(dummy.x + 4) < 1e-4


@pytest.mark.visualization
def test_gss_optimizer_nlin_fcn():
    from matplotlib import pyplot as plt

    # Parameters
    x_grid = to.linspace(-2.0, 3.0, 200)
    f = 1.0
    noise_std = 0.1

    # Init param and optimizer
    x_init = (
        to.rand(1) * (x_grid.max() - x_grid.min()) / 2 + x_grid.min() + (x_grid.max() - x_grid.min()) / 4
    )  # [.25, .75]
    x = nn.Parameter(to.tensor([x_init]), requires_grad=False)
    optim = GSS([x], param_min=x_grid.min().unsqueeze(0), param_max=x_grid.max().unsqueeze(0))
    obj_fcn = partial(noisy_nonlin_fcn, x=x, f=f, noise_std=noise_std)
    num_epochs = 10

    # Init plotting
    plt.figure()
    plt.plot(x_grid, noisy_nonlin_fcn(x=x_grid, f=f), label="noise free fcn")
    plt.scatter(x.data.numpy(), obj_fcn().numpy(), s=40, marker="x", color="k", label="init guess")
    colors = plt.get_cmap("inferno")(np.linspace(0, 1, num_epochs))

    for e in tqdm(range(num_epochs), total=num_epochs):
        # Evaluate at a the current point
        optim.step(obj_fcn)

        # Plot current evaluation
        plt.plot(x_grid, noisy_nonlin_fcn(x=x_grid, f=f, noise_std=noise_std), alpha=0.2)
        plt.scatter(x.data.numpy(), obj_fcn().numpy(), s=16, color=colors[e])

    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend()
    plt.show()
    assert noisy_nonlin_fcn(x, f=f, noise_std=noise_std) < noisy_nonlin_fcn(x_init, f=f, noise_std=noise_std)


@pytest.mark.parametrize("dt", [0.1], ids=["0.1"])
@pytest.mark.parametrize("t_end", [6.0], ids=["1."])
@pytest.mark.parametrize(
    "t_intvl_space",
    [
        BoxSpace(0.1, 0.11, shape=1),
        BoxSpace(0.123, 0.456, shape=1),
        BoxSpace(10.0, 20.0, shape=1),
    ],
    ids=["small_time_intvl", "real_time_intvl", "large_time_intvl"],
)
@pytest.mark.parametrize("val_space", [BoxSpace(-5.0, 3.0, shape=1)], ids=["-5_to_3"])
def test_skyline(dt: Union[int, float], t_end: Union[int, float], t_intvl_space: BoxSpace, val_space: BoxSpace):
    # Create the skyline function
    t, vals = skyline(dt, t_end, t_intvl_space, val_space)
    assert isinstance(t, np.ndarray) and isinstance(vals, np.ndarray)
    assert len(t) == len(vals)


def test_check_prompt():
    with completion_context("Works fine", color="g"):
        a = 3
    with pytest.raises(ZeroDivisionError):
        with completion_context("Works fine", color="r", bright=True):
            a = 3 / 0


@pytest.mark.parametrize("color", ["w", "g", "y", "c", "r", "b"], ids=["w", "g", "y", "c", "r", "b"])
@pytest.mark.parametrize("bright", [True, False], ids=["bright", "not_bright"])
def test_print_cbt_once(color, bright):
    # Reset the flag for this test
    print_cbt_once.has_run = False

    msg = "You should only read this once per color and brightness"
    for i in range(10):
        print_cbt_once(msg, color, bright, tag="tag", end="\n")
        if i > 0:
            assert print_cbt_once.has_run


@pytest.mark.parametrize("x", [to.rand(5, 3), np.random.rand(5, 3)], ids=["torch", "numpy"])
@pytest.mark.parametrize("dim", [0, 1], ids=["dim0", "dim1"])
def test_logmeanexp(x, dim):
    lme = logmeanexp(x, dim)
    assert lme is not None
    if isinstance(x, to.Tensor) and isinstance(lme, to.Tensor):
        assert to.allclose(lme, to.log(to.mean(to.exp(x), dim=dim)))
    if isinstance(x, np.ndarray):
        assert np.allclose(lme, np.log(np.mean(np.exp(x), axis=dim)))


@pytest.mark.parametrize(
    "obj, file_ext",
    [
        (to.rand((5, 3)), "pt"),
        (np.random.rand(5, 3), "npy"),
        (DummyPolicy(BallOnBeamSim(dt=0.01, max_steps=500).spec), "pt"),
        (BallOnBeamSim(dt=0.01, max_steps=500), "pkl"),
    ],
    ids=["tensor", "ndarray", "dummypol", "pyenv"],
)
@pytest.mark.parametrize("prefix", ["", "pre"], ids=["defualt", "pre"])
@pytest.mark.parametrize("suffix", ["", "suf"], ids=["defualt", "suf"])
@pytest.mark.parametrize("use_state_dict", [True, False], ids=["use_state_dict", "not-use_state_dict"])
@pytest.mark.parametrize("verbose", [True, False], ids=["v", "q"])
def test_save_load(obj, file_ext: str, tmpdir, prefix: str, suffix: str, use_state_dict: bool, verbose: bool):
    # Save
    pyrado.save(obj, f"tmpname.{file_ext}", tmpdir, prefix, suffix, use_state_dict)

    # Check if sth has been saved with the correct pre- and suffix
    if prefix == "" and suffix == "":
        assert osp.exists(osp.join(tmpdir, f"tmpname.{file_ext}"))
    elif prefix != "" and suffix != "":
        assert osp.exists(osp.join(tmpdir, f"{prefix}_tmpname_{suffix}.{file_ext}"))
    elif prefix != "" and suffix == "":
        assert osp.exists(osp.join(tmpdir, f"{prefix}_tmpname.{file_ext}"))
    elif prefix == "" and suffix != "":
        assert osp.exists(osp.join(tmpdir, f"tmpname_{suffix}.{file_ext}"))

    # Check if sth has been loaded with the correct pre- and suffix
    res = pyrado.load(f"tmpname.{file_ext}", tmpdir, prefix, suffix, obj, verbose)
    assert res is not None


@pytest.mark.parametrize(
    "s",
    [[-1, 0, 1], np.array([[-1, 0, 1]]), np.array([-4, -2, -1, 0])],
)
@pytest.mark.parametrize(
    "d",
    [1, 2],
)
@pytest.mark.parametrize(
    "h",
    [1, 1e-4],
)
def test_diff_coeffs(s, d, h):
    coeffs, order = numerical_differentiation_coeffs(s, d, h)
    assert sum(coeffs) == 0
    assert order > 0


@pytest.mark.parametrize(
    "x, b",
    [
        [iter((1, 2, 3)), True],
        [iter([1, 2, 3]), True],
        [iter(dict(a=1, b=2)), True],
        [(1, 2, 3), False],
        [[1, 2, 3], False],
    ],
)
def test_is_iterator(x, b):
    assert is_iterator(x) == b


@pytest.mark.parametrize(
    "x, b",
    [
        [(1, 2, 3), True],
        [[1, 2, 3], True],
        [dict(a=1, b=2), True],
        [1, False],
    ],
)
def test_is_sequence(x, b):
    assert is_sequence(x) == b


@pytest.mark.parametrize("dtype", ["torch", "numpy"], ids=["to", "np"])
@pytest.mark.parametrize(
    "lb, ub",
    [(0, 1), (-1, 1), (-2.5, 0), (-np.ones((3, 2)), np.ones((3, 2)))],
    ids=["lb0_ub1", "lb-1_ub1", "lb-2.5_ub0", "np_ones"],
)
def test_minmaxscaler(dtype, lb, ub):
    for _ in range(10):
        scaler = MinMaxScaler(lb, ub)
        x = 1e2 * to.rand(3, 2) if dtype == "torch" else np.random.rand(3, 2)
        x_scaled = scaler.scale_to(x)
        x_scaled_back = scaler.scale_back(x_scaled)

        if isinstance(x_scaled, to.Tensor):
            x_scaled = x_scaled.numpy()  # for easier checking with pytest.approx
            x_scaled_back = x_scaled_back.numpy()  # for easier checking with pytest.approx
        assert np.all(scaler._bound_lo * np.ones_like(x_scaled) <= x_scaled)
        assert np.all(x_scaled <= scaler._bound_up * np.ones_like(x_scaled))
        assert np.allclose(x, x_scaled_back)


@pytest.mark.parametrize(
    "x, b",
    [
        [(1, 2, 3), True],
        [(1.0, 2.0, 3.0), True],
        [[1, 2, 3], True],
        [(1, 2, 3.0), False],
        [(1, 2, [3]), False],
    ],
)
def test_check_all_types_equal(x, b):
    assert check_all_types_equal(x) == b


@pytest.mark.parametrize(
    "x, b",
    [
        [([1], [2], [3]), True],
        [([1, 2], [2], [3]), False],
        [(np.array([1]), np.array([2]), np.array([3])), True],
        [(np.array([1, 2]), np.array([2]), np.array([3])), False],
        [(to.tensor([1]), to.tensor([2]), to.tensor([3])), True],
        [(to.tensor([1, 2]), to.tensor([2]), to.tensor([3])), False],
    ],
)
def test_check_all_lengths_equal(x, b):
    assert check_all_lengths_equal(x) == b


@pytest.mark.parametrize(
    "x, b",
    [
        [(np.array([1]), np.array([2]), np.array([3])), True],
        [(np.array([1, 2]), np.array([2]), np.array([3])), False],
        [(to.tensor([1]), to.tensor([2]), to.tensor([3])), True],
        [(to.tensor([1, 2]), to.tensor([2]), to.tensor([3])), False],
    ],
)
def test_check_all_shapes_equal(x, b):
    assert check_all_shapes_equal(x) == b


def test_iteration_tracker():
    tracker: IterationTracker = IterationTracker()
    assert isinstance(tracker, IterationTracker)
    with pytest.raises(IndexError):
        tracker.pop()
    tracker.push("meta", 1)
    tracker.push("sub", 42)
    assert str(tracker) == "meta_1-sub_42"
    assert tracker.peek() == ("sub", 42)
    assert list(tracker) == [("meta", 1), ("sub", 42)]
    assert tracker.pop() == ("sub", 42)
    assert tracker.pop() == ("meta", 1)
    assert tracker.get("magic") is None
    with tracker.iteration("meta", 1):
        assert tracker.peek() == ("meta", 1)
        with tracker.iteration("sub", 42):
            assert tracker.get("sub") == 42
            assert tracker.format() == "meta_1-sub_42"


@pytest.mark.parametrize("x", [np.empty((7,)), np.empty((7, 1)), to.empty((7,)), to.empty((7, 1))])
def test_correct_atleast_2d(x):
    x_corrected = correct_atleast_2d(x)
    assert x_corrected.shape[0] == len(x)
