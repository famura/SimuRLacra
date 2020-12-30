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
import random
import time
from torch.distributions.multivariate_normal import MultivariateNormal

import pyrado
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.base import Policy
from pyrado.sampling.data_format import to_format
from pyrado.sampling.hyper_sphere import sample_from_hyper_sphere_surface
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.parameter_exploration_sampler import ParameterExplorationSampler
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.sampler_pool import *
from pyrado.sampling.sequences import *
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.policies.features import *
from pyrado.sampling.cvar_sampler import select_cvar
from pyrado.utils.data_types import RenderMode
from tests.conftest import m_needs_cuda, m_needs_bullet


@pytest.mark.parametrize(
    "arg",
    [
        [1],
        [2, 3],
        [4, 6, 2, 88, 3, 45, 7, 21, 22, 23, 24, 44, 45, 56, 67, 78, 89],
    ],
)
def test_sampler_pool(arg):
    pool = SamplerPool(len(arg))
    result = pool.invoke_all_map(_cb_test_eachhandler, arg)
    pool.stop()

    assert result == list(map(lambda x: x * 2, arg))


def _cb_test_eachhandler(G, arg):
    time.sleep(random.randint(1, 5))
    return arg * 2


def _cb_test_collecthandler(G):
    nsample = random.randint(5, 15)
    return nsample, nsample


@pytest.mark.parametrize("num_threads", [1, 2, 4])
@pytest.mark.parametrize("min_samples", [10, 20, 40])
def test_sampler_collect(num_threads: int, min_samples: int):
    pool = SamplerPool(num_threads)

    # Run the collector
    cr, cn = pool.run_collect(min_samples, _cb_test_collecthandler)
    pool.stop()

    assert min_samples <= cn
    assert min_samples <= sum(cr)


@pytest.mark.parametrize("num_threads", [1, 2, 4])
@pytest.mark.parametrize("min_samples", [10, 20, 40])
@pytest.mark.parametrize("min_runs", [10, 20, 40])
def test_sampler_collect_minrun(num_threads: int, min_samples: int, min_runs: int):
    pool = SamplerPool(num_threads)

    # Run the collector
    cr, cn = pool.run_collect(min_samples, _cb_test_collecthandler, min_runs=min_runs)

    pool.stop()

    assert min_samples <= cn
    assert min_samples <= sum(cr)
    assert min_runs <= len(cr)


@pytest.mark.parametrize(
    "data_type",
    [
        (None, None),
        (to.int32, np.int32),
    ],
)
def test_to_format(data_type: tuple):
    # Create some tensors to convert
    ndarray = np.random.rand(3, 2).astype(dtype=np.float64)
    tensor = to.rand(3, 2).type(dtype=to.float64)

    # Test the conversion and typing from numpy to PyTorch
    converted_ndarray = to_format(ndarray, "torch", data_type[0])
    assert isinstance(converted_ndarray, to.Tensor)
    new_type = to.float64 if data_type[0] is None else data_type[0]  # passing None must not change the type
    assert converted_ndarray.dtype == new_type

    # Test the conversion and typing from PyTorch to numpy
    converted_tensor = to_format(tensor, "numpy", data_type[1])
    assert isinstance(converted_tensor, np.ndarray)
    new_type = np.float64 if data_type[1] is None else data_type[1]  # passing None must not change the type
    assert converted_tensor.dtype == new_type


@pytest.mark.parametrize(
    "epsilon",
    [
        1,
        0.5,
        0.1,
    ],
)
@pytest.mark.parametrize(
    "num_ro",
    [
        10,
        20,
    ],
)
def test_select_cvar(epsilon: float, num_ro: int):
    # Create rollouts with known discounted rewards
    rollouts = [StepSequence(rewards=[i], observations=[i], actions=[i]) for i in range(num_ro)]
    # Shuffle data to put in
    ro_shuf = list(rollouts)
    random.shuffle(ro_shuf)

    # Select cvar quantile
    ro_cv = select_cvar(ro_shuf, epsilon, 1)

    # Compute expected return of subselection
    cv = sum(map(lambda ro: ro.discounted_return(1), ro_cv)) / len(ro_cv)

    # This should be equal to the epsilon-quantile of the integer sequence
    nq = int(num_ro * epsilon)
    cv_expected = sum(range(nq)) / nq

    assert cv == cv_expected


@pytest.mark.parametrize(
    "num_dim, method",
    [
        (1, "uniform"),
        (1, "uniform"),
        (3, "uniform"),
        (3, "normal"),
        (3, "Marsaglia"),
        (4, "uniform"),
        (4, "normal"),
        (4, "Marsaglia"),
        (15, "uniform"),
        (15, "normal"),
    ],
)
def test_sample_from_unit_sphere_surface(num_dim: int, method: str):
    s = sample_from_hyper_sphere_surface(num_dim, method)
    assert 0.95 <= to.norm(s, p=2) <= 1.05


@pytest.mark.parametrize(
    ["env", "policy"],
    [
        ("default_bob", "idle_policy"),
        ("default_bob", "dummy_policy"),
        ("default_bob", "time_policy"),
        ("default_bob", "linear_policy"),
        ("default_bob", "fnn_policy"),
        ("default_bob", "rnn_policy"),
        ("default_bob", "lstm_policy"),
        ("default_bob", "gru_policy"),
        ("default_bob", "adn_policy"),
        ("default_bob", "nf_policy"),
        ("default_bob", "thfnn_policy"),
        ("default_bob", "thgru_policy"),
    ],
    ids=[
        "bob_idle",
        "bob_dummy",
        "bob_time",
        "bob_lin",
        "bob_fnn",
        "bob_rnn",
        "bob_lstm",
        "bob_gru",
        "bob_adn",
        "bob_nf",
        "bob_thfnn",
        "bob_thgru",
    ],
    indirect=True,
)
def test_rollout_wo_exploration(env: SimEnv, policy: Policy):
    ro = rollout(env, policy, render_mode=RenderMode())
    assert isinstance(ro, StepSequence)
    assert len(ro) <= env.max_steps


@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
def test_rollout_wo_policy(env: SimEnv):
    def policy(obs):
        # Callable must receive and return tensors
        return to.from_numpy(env.spec.act_space.sample_uniform())

    ro = rollout(env, policy, render_mode=RenderMode())
    assert isinstance(ro, StepSequence)
    assert len(ro) <= env.max_steps


@pytest.mark.parametrize(
    "mean, cov",
    [
        (to.tensor([5.0, 7.0]), to.tensor([[2.0, 0.0], [0.0, 2.0]])),
    ],
    ids=["2dim"],
)
def test_reparametrization_trick(mean, cov):
    for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # Sampling the the PyTorch distribution class
        distr_mvn = MultivariateNormal(mean, cov)
        to.manual_seed(seed)
        smpl_distr = distr_mvn.sample()

        # The reparametrization trick done by PyTorch
        to.manual_seed(seed)
        smpl_distr_reparam = distr_mvn.sample()

        # The reparametrization trick done by hand
        to.manual_seed(seed)
        smpl_reparam = mean + to.cholesky(cov, upper=False).mv(to.randn_like(mean))

        to.testing.assert_allclose(smpl_distr, smpl_distr_reparam)
        to.testing.assert_allclose(smpl_distr, smpl_reparam)
        to.testing.assert_allclose(smpl_distr_reparam, smpl_reparam)


# @pytest.mark.visualization
@pytest.mark.parametrize(
    "sequence, x_init",
    [
        (sequence_const, np.array([2])),
        (sequence_plus_one, np.array([2])),
        (sequence_add_init, np.array([2])),
        (sequence_rec_double, np.array([2])),
        (sequence_rec_sqrt, np.array([2])),
        (sequence_nlog2, np.array([2])),
        (sequence_const, np.array([1, 2, 3])),
        (sequence_plus_one, np.array([1, 2, 3])),
        (sequence_add_init, np.array([1, 2, 3])),
        (sequence_rec_double, np.array([1, 2, 3])),
        (sequence_rec_sqrt, np.array([1, 2, 3])),
        (sequence_nlog2, np.array([1, 2, 3])),
    ],
)
def test_sequences(sequence: Callable, x_init: np.ndarray):
    # Get the full sequence
    _, x_full = sequence(x_init, 5, float)
    assert x_full is not None

    # Plot the sequences
    # for i in range(x_full.shape[1]):
    #     plt.stem(x_full[:, i], label=str(x_init[i]))
    # plt.legend()
    # plt.show()


@pytest.mark.parametrize("sample", [np.array([30, 37, 36, 43, 42, 43, 43, 46, 41, 42])])
@pytest.mark.parametrize("seed", [1, 12, 123], ids=["seed1", "seed1", "seed123"])
def test_boostrap_methods(sample, seed):
    # Emperical bootstrap
    m_bs, ci_bs_lo, ci_bs_up = bootstrap_ci(sample, np.mean, num_reps=20, alpha=0.1, ci_sides=2, seed=seed)

    # Percentile bootstrap
    pyrado.set_seed(seed)
    resampled = np.random.choice(sample, (sample.shape[0], 20), replace=True)
    means = np.apply_along_axis(np.mean, 0, resampled)
    ci_lo, ci_up = np.percentile(means, [5, 95])

    # You should operate on the deltas (emperical bootsrap) and not directly on the statistic from the resampled data
    # (percentile bootsrap)
    assert ci_lo != ci_bs_lo
    assert ci_up != ci_bs_up


@pytest.mark.parametrize(
    "data",
    [np.random.normal(10, 1, (40,)), np.random.normal((1, 7, 13), (1, 1, 1), (40, 3))],
    ids=["1dim-data", "3dim-data"],
)
@pytest.mark.parametrize("num_reps", [100, 1000, 10000], ids=["100reps", "1000reps", "10000reps"])
@pytest.mark.parametrize("seed", [1, 12, 123], ids=["seed1", "seed12", "seed123"])
def test_bootsrapping(data, num_reps, seed):
    # Fully-fledged example
    bootstrap_ci(data, np.mean, num_reps, alpha=0.05, ci_sides=2, studentized=True, bias_correction=True, seed=seed)

    m, ci_lo, ci_up = bootstrap_ci(
        data, np.mean, num_reps, alpha=0.05, ci_sides=2, studentized=False, bias_correction=False, seed=seed
    )
    assert np.all(m >= ci_lo)
    assert np.all(m <= ci_up)

    m_bc, ci_lo, ci_up = bootstrap_ci(
        data, np.mean, num_reps, alpha=0.05, ci_sides=2, studentized=False, bias_correction=True, seed=seed
    )
    assert np.all(m_bc != m)

    m, ci_lo, ci_up = bootstrap_ci(data, np.mean, num_reps, alpha=0.05, ci_sides=1, studentized=False, seed=seed)
    m_t, ci_lo_t, ci_up_t = bootstrap_ci(data, np.mean, num_reps, alpha=0.05, ci_sides=1, studentized=True, seed=seed)
    assert m == pytest.approx(m_t)
    assert np.all(m_t >= ci_lo_t)
    assert np.all(m_t <= ci_up_t)
    # Bounds are different (not generally wider) when assuming a t-distribution
    assert np.all(ci_lo != ci_lo_t)
    assert np.all(ci_up != ci_up_t)


@pytest.mark.parametrize(
    ["env", "policy"],
    [
        ("default_bob", "fnn_policy"),
    ],
    ids=["bob_fnnpol"],
    indirect=True,
)
@pytest.mark.parametrize(
    ["num_rollouts_per_param", "fixed_init_state"],
    [
        (1, False),
        (1, True),
        (9, False),
        (9, True),
    ],
    ids=["1rops-randinit", "1rops-fixedinit", "9rops-randinit", "9rops-fixedinit"],
)
def test_param_expl_sampler(env: SimEnv, policy: Policy, num_rollouts_per_param: int, fixed_init_state: bool):
    # Add randomizer
    pert = create_default_randomizer(env)
    env = DomainRandWrapperLive(env, pert)

    # Create the sampler
    sampler = ParameterExplorationSampler(env, policy, num_rollouts_per_param, num_workers=1)

    # Use some random parameters
    num_ps = 7
    params = to.rand(num_ps, policy.num_param)

    if fixed_init_state:
        # Sample a custom init state
        init_states = [env.init_space.sample_uniform()] * num_rollouts_per_param
    else:
        # Let the sampler forward to the env to randomly sample an init state
        init_states = None

    # Do the sampling
    samples = sampler.sample(param_sets=params, init_states=init_states)

    # Check if the correct number of rollouts has been sampled
    assert num_ps == len(samples)
    assert num_ps * num_rollouts_per_param == samples.num_rollouts
    for ps in samples:
        assert len(ps.rollouts) == num_rollouts_per_param

    # Compare rollouts that should be matching
    for idx in range(num_rollouts_per_param):
        # Use the first parameter set as pivot
        piter = iter(samples)
        pivot = next(piter).rollouts[idx]

        # Iterate through others
        for ops in piter:
            other_ro = ops.rollouts[idx]
            # Compare domain params
            assert pivot.rollout_info["domain_param"] == other_ro.rollout_info["domain_param"]
            # Compare first observation a.k.a. init state
            assert pivot[0].observation == pytest.approx(other_ro[0].observation)


@m_needs_cuda
@pytest.mark.wrapper
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qbb",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    [
        "fnn_policy",
        "lstm_policy",
    ],
    ids=["fnn", "lstm"],
    indirect=True,
)
def test_cuda_sampling_w_dr(env: SimEnv, policy: Policy):
    randomizer = create_default_randomizer(env)
    env = DomainRandWrapperLive(env, randomizer)

    sampler = ParallelRolloutSampler(env, policy, num_workers=2, min_rollouts=10)
    samples = sampler.sample()

    assert samples is not None


@pytest.mark.parametrize(
    "env",
    [
        pytest.param("default_qqsurcs_bt", marks=m_needs_bullet),
        pytest.param("default_bs_ds_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bit_ika_pos_bt", marks=m_needs_bullet),
    ],
    ids=["qqsurcs_bt", "bs_pos_bt", "bit_ika_bt"],
    indirect=True,
)
@pytest.mark.parametrize("policy", ["idle_policy"], ids=["idle"], indirect=True)  # ad deterministic policy
@pytest.mark.parametrize(
    "num_simulations", [1, 4], ids=["1sim", "4sims"]  # must be lower than the number of cores on the machine
)
def test_sequential_equals_parallel(env: SimEnv, policy: Policy, num_simulations: int):
    # Do the rollouts explicitly sequentially without a sampler
    # Do not set the init state to check if this was sampled correctly
    ros_sequential = []
    for i in range(num_simulations):
        ros_sequential.append(rollout(env, policy, eval=True, seed=i))

    # Do the rollouts in parallel with a sampler. Create one worker for every rollout
    # Do not set the init state to check if this was sampled correctly
    sampler = ParallelRolloutSampler(env, policy, num_workers=num_simulations, min_rollouts=num_simulations, seed=0)
    ros_parallel = sampler.sample()
    assert len(ros_parallel) == num_simulations

    for ro_s in ros_sequential:
        # The parallel rollouts are not necessarily in the same order as the sequential ones, thus compare to all
        assert any([ro_s.observations == pytest.approx(ro_p.observations) for ro_p in ros_parallel])
