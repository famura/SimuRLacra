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

import random
import time
from typing import Optional

import pytest
from tests.conftest import m_needs_bullet, m_needs_cuda
from torch.distributions.multivariate_normal import MultivariateNormal

from pyrado.algorithms.step_based.a2c import A2C
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.algorithms.utils import RolloutSavingWrapper
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.sim_base import SimEnv
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.logger import set_log_prefix_dir
from pyrado.policies.base import Policy
from pyrado.policies.features import *
from pyrado.policies.feed_back.fnn import FNN
from pyrado.policies.feed_forward.dummy import IdlePolicy
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.sampling.cvar_sampler import select_cvar
from pyrado.sampling.data_format import to_format
from pyrado.sampling.hyper_sphere import sample_from_hyper_sphere_surface
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.sampling.parameter_exploration_sampler import ParameterExplorationSampler, ParameterSamplingResult
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sampler_pool import *
from pyrado.sampling.sequences import *
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode


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


def _cb_test_collecthandler(G, num):
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


@pytest.mark.parametrize("data_type", [(None, None), (to.int32, np.int32)])
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


@pytest.mark.parametrize("epsilon", [1, 0.5, 0.1])
@pytest.mark.parametrize("num_ro", [10, 20])
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
        ("default_bob", "pst_policy"),
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
        "bob_pst",
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
    [(to.tensor([5.0, 7.0]), to.tensor([[2.0, 0.0], [0.0, 2.0]]))],
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
    ["num_init_states_per_domain", "fixed_init_state", "num_domains"],
    [
        (1, False, 1),
        (1, True, 1),
        (9, False, 1),
        (9, True, 1),
    ],
    ids=["1rops-randinit", "1rops-fixedinit", "9rops-randinit", "9rops-fixedinit"],
)
@pytest.mark.parametrize("num_workers", [1, 4], ids=["1worker", "4workers"])
def test_param_expl_sampler(
    env: SimEnv,
    policy: Policy,
    num_init_states_per_domain: int,
    fixed_init_state: bool,
    num_domains: int,
    num_workers: int,
):
    num_rollouts_per_param = num_init_states_per_domain * num_domains

    # Add randomizer
    pert = create_default_randomizer(env)
    env = DomainRandWrapperLive(env, pert)

    # Create the sampler
    sampler = ParameterExplorationSampler(env, policy, num_init_states_per_domain, num_domains, num_workers=num_workers)

    # Use some random parameters
    num_ps = 7
    params = to.rand(num_ps, policy.num_param)

    if fixed_init_state:
        # Sample a custom init state
        init_states = [env.init_space.sample_uniform()] * num_init_states_per_domain
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


@pytest.mark.parametrize("env", ["default_bob"], indirect=True, ids=["bob"])
@pytest.mark.parametrize(
    "policy",
    [
        "linear_policy",
        "fnn_policy",
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
        "thfnn_policy",
        "thgru_policy",
    ],
    ids=["lin", "fnn", "rnn", "lstm", "gru", "adn", "nf", "thfnn", "thgru"],
    indirect=True,
)
@pytest.mark.parametrize("num_workers", [1, 4], ids=["1worker", "4workers"])
def test_parameter_exploration_sampler(env: SimEnv, policy: Policy, num_workers: int):
    # Use some random parameters
    num_ps = 7
    params = to.rand(num_ps, policy.num_param)

    sampler = ParameterExplorationSampler(
        env, policy, num_init_states_per_domain=1, num_domains=1, num_workers=num_workers
    )
    psr = sampler.sample(param_sets=params)
    assert isinstance(psr, ParameterSamplingResult)
    assert len(psr.rollouts) >= 1 * 1 * num_ps


@pytest.mark.parametrize("policy", ["dummy_policy", "idle_policy"], ids=["dummy", "idle"], indirect=True)
@pytest.mark.parametrize("env", ["default_qbb"], ids=["qbb"], indirect=True)
@pytest.mark.parametrize("num_params", [2])
@pytest.mark.parametrize("num_init_states_per_domain", [2])
@pytest.mark.parametrize("num_domains", [2])
@pytest.mark.parametrize("set_init_states", [False, True], ids=["wo_init_states", "w_init_states"])
def test_parameter_exploration_sampler_deterministic(
    env: SimEnv,
    policy: Policy,
    num_params: int,
    num_init_states_per_domain: int,
    num_domains: int,
    set_init_states: bool,
):
    param_sets = to.rand(num_params, policy.num_param)

    if set_init_states:
        init_states = [env.spec.state_space.sample_uniform() for _ in range(num_init_states_per_domain * num_domains)]
    else:
        init_states = None

    nums_workers = (1, 2, 4)

    all_results = []
    for num_workers in nums_workers:
        # Reset the seed every time because sample() uses the root sampler. This does not matter for regular runs, but
        # for this tests it is very relevant.
        pyrado.set_seed(0)
        all_results.append(
            ParameterExplorationSampler(
                env,
                policy,
                num_init_states_per_domain=num_init_states_per_domain,
                num_domains=num_domains,
                num_workers=num_workers,
                seed=0,
            ).sample(param_sets=param_sets, init_states=init_states)
        )

    # Test that the rollouts for all number of workers are equal.
    for psr_a, psr_b in [(a, b) for a in all_results for b in all_results]:
        assert psr_a.parameters == pytest.approx(psr_b.parameters)
        assert psr_a.mean_returns == pytest.approx(psr_b.mean_returns)
        assert psr_a.num_rollouts == psr_b.num_rollouts
        assert len(psr_a.rollouts) == len(psr_b.rollouts)
        for ros_a, ros_b in zip(psr_a.rollouts, psr_b.rollouts):
            for ro_a, ro_b in zip(ros_a, ros_b):
                assert ro_a.rewards == pytest.approx(ro_b.rewards)
                assert ro_a.observations == pytest.approx(ro_b.observations)
                assert ro_a.actions == pytest.approx(ro_b.actions)


@pytest.mark.parametrize("env", ["default_bob"], indirect=True, ids=["bob"])
@pytest.mark.parametrize(
    "policy",
    [
        "linear_policy",
        "fnn_policy",
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
        "thfnn_policy",
        "thgru_policy",
    ],
    ids=["lin", "fnn", "rnn", "lstm", "gru", "adn", "nf", "thfnn", "thgru"],
    indirect=True,
)
@pytest.mark.parametrize("num_workers", [1, 4], ids=["1worker", "4workers"])
def test_parallel_rollout_sampler(env: SimEnv, policy: Policy, num_workers: int):
    min_rollouts = num_workers * 2  # make sure every worker samples at least once
    sampler = ParallelRolloutSampler(env, policy, num_workers, min_rollouts=min_rollouts)
    ros = sampler.sample()
    assert isinstance(ros, list)
    assert len(ros) >= min_rollouts


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
        "fnn_policy_cuda",
        "lstm_policy",
        "lstm_policy_cuda",
    ],
    ids=["fnn", "fnn_cuda", "lstm", "lstm_cuda"],
    indirect=True,
)
@pytest.mark.parametrize("num_workers", [1, 2], ids=["1worker", "4workers"])
def test_cuda_sampling_w_dr(env: SimEnv, policy: Policy, num_workers: int):
    randomizer = create_default_randomizer(env)
    env = DomainRandWrapperLive(env, randomizer)

    sampler = ParallelRolloutSampler(env, policy, num_workers=num_workers, min_rollouts=4)
    samples = sampler.sample()

    assert samples is not None


@pytest.mark.parametrize(
    "env",
    [
        "default_pend",
        "default_qbb",
        pytest.param("default_qqsurcs_bt", marks=m_needs_bullet),
    ],
    ids=["pend", "qbb", "qqsurcs_bt"],
    indirect=True,
)
@pytest.mark.parametrize("policy", ["dummy_policy", "idle_policy"], ids=["dummy", "idle"], indirect=True)
@pytest.mark.parametrize("num_rollouts", [1, 2, 4, 6])
@pytest.mark.parametrize("num_workers", [1, 2, 4])
def test_sequential_equals_parallel(env: SimEnv, policy: Policy, num_rollouts: int, num_workers: int):
    # Do the rollouts explicitly sequentially without a sampler.
    # Do not set the init state to check if this was sampled correctly.
    ros_sequential = []
    for i in range(num_rollouts):
        ros_sequential.append(rollout(env, policy, eval=True, seed=0, sub_seed=0, sub_sub_seed=i))

    # Do the rollouts in parallel with a sampler.
    # Do not set the init state to check if this was sampled correctly.
    sampler = ParallelRolloutSampler(env, policy, num_workers=num_workers, min_rollouts=num_rollouts, seed=0)
    ros_parallel = sampler.sample()
    assert len(ros_parallel) == num_rollouts

    for ro_s, ro_p in zip(ros_sequential, ros_parallel):
        assert ro_s.rewards == pytest.approx(ro_p.rewards)
        assert ro_s.observations == pytest.approx(ro_p.observations)
        assert ro_s.actions == pytest.approx(ro_p.actions)


@pytest.mark.parametrize("policy", ["dummy_policy", "fnn_policy"], ids=["dummy", "fnn"], indirect=True)
@pytest.mark.parametrize("env", ["default_qbb"], ids=["qbb"], indirect=True)
@pytest.mark.parametrize("min_rollouts", [2, 4, 6])  # Once less, equal, and more rollouts than workers.
@pytest.mark.parametrize("init_states", [None, 2])
@pytest.mark.parametrize("domain_params", [None, [{"g": 10}]])
def test_parallel_sampling_deterministic_wo_min_steps(
    env: SimEnv,
    policy: Policy,
    min_rollouts: Optional[int],
    init_states: Optional[int],
    domain_params: Optional[List[dict]],
):
    if init_states is not None:
        init_states = [env.spec.state_space.sample_uniform() for _ in range(init_states)]

    nums_workers = (1, 2, 4)

    all_rollouts = []
    for num_workers in nums_workers:
        # Act an exploration strategy to test if that works too (it should as the policy gets pickled and distributed
        # anyway).
        all_rollouts.append(
            ParallelRolloutSampler(
                env,
                NormalActNoiseExplStrat(policy, std_init=1.0),
                num_workers=num_workers,
                min_rollouts=min_rollouts,
                seed=0,
            ).sample(init_states=init_states, domain_params=domain_params)
        )

    # Test that the rollouts are actually different, i.e., that not the same seed is used for all rollouts.
    for ros in all_rollouts:
        for ro_a, ro_b in [(a, b) for a in ros for b in ros if a is not b]:
            # The idle policy iy deterministic and always outputs the zero action. Hence, do not check that the actions
            # are different when using the idle policy.
            if isinstance(policy, IdlePolicy):
                # The Quanser Ball Balancer is a deterministic environment (conditioned on the initial state). As the
                # idle policy is a deterministic policy, this will result in the rollouts being equivalent for each
                # initial state, so do not check for difference if the initial states where set.
                if init_states is None:
                    assert ro_a.rewards != pytest.approx(ro_b.rewards)
                    assert ro_a.observations != pytest.approx(ro_b.observations)
            else:
                assert ro_a.rewards != pytest.approx(ro_b.rewards)
                assert ro_a.observations != pytest.approx(ro_b.observations)
                assert ro_a.actions != pytest.approx(ro_b.actions)

    # Test that the rollouts for all number of workers are equal.
    for ros_a, ros_b in [(a, b) for a in all_rollouts for b in all_rollouts]:
        assert len(ros_a) == len(ros_b)
        for ro_a, ro_b in zip(ros_a, ros_b):
            assert ro_a.rewards == pytest.approx(ro_b.rewards)
            assert ro_a.observations == pytest.approx(ro_b.observations)
            assert ro_a.actions == pytest.approx(ro_b.actions)


# Include the FNN policy as it requires initialization which also has to be seeded.
@pytest.mark.parametrize("policy", ["dummy_policy", "fnn_policy"], ids=["dummy", "fnn"], indirect=True)
@pytest.mark.parametrize("env", ["default_qbb"], ids=["qbb"], indirect=True)
@pytest.mark.parametrize("min_rollouts", [None, 2, 4, 6])  # Once less, equal, and more rollouts than workers.
@pytest.mark.parametrize("min_steps", [2, 10, 30])
@pytest.mark.parametrize("domain_params", [None, [{"g": 10}]])
def test_parallel_sampling_deterministic_w_min_steps(
    env: SimEnv,
    policy: Policy,
    min_rollouts: Optional[int],
    min_steps: int,
    domain_params: Optional[List[dict]],
):
    nums_workers = (1, 2, 4)

    all_rollouts = []
    for num_workers in nums_workers:
        # Act an exploration strategy to test if that works too (it should as the policy gets pickled and distributed
        # anyway).
        all_rollouts.append(
            ParallelRolloutSampler(
                env,
                NormalActNoiseExplStrat(policy, std_init=1.0),
                num_workers=num_workers,
                min_rollouts=min_rollouts,
                min_steps=min_steps * env.max_steps,
                seed=0,
            ).sample(domain_params=domain_params)
        )

    # Test that the rollouts are actually different, i.e., that not the same seed is used for all rollouts.
    for ros in all_rollouts:
        for ro_a, ro_b in [(a, b) for a in ros for b in ros if a is not b]:
            # The idle policy iy deterministic and always outputs the zero action. Hence, do not check that the actions
            # are different when using the idle policy.
            if not isinstance(policy, IdlePolicy):
                assert ro_a.rewards != pytest.approx(ro_b.rewards)
                assert ro_a.observations != pytest.approx(ro_b.observations)
                assert ro_a.actions != pytest.approx(ro_b.actions)

    # Test that the rollouts for all number of workers are equal.
    for ros_a, ros_b in [(a, b) for a in all_rollouts for b in all_rollouts]:
        assert sum([len(ro) for ro in ros_a]) >= min_steps * env.max_steps
        assert sum([len(ro) for ro in ros_b]) >= min_steps * env.max_steps
        assert sum([len(ro) for ro in ros_a]) == sum([len(ro) for ro in ros_b])
        assert len(ros_a) == len(ros_b)
        for ro_a, ro_b in zip(ros_a, ros_b):
            assert ro_a.rewards == pytest.approx(ro_b.rewards)
            assert ro_a.observations == pytest.approx(ro_b.observations)
            assert ro_a.actions == pytest.approx(ro_b.actions)


@pytest.mark.parametrize("env", ["default_qbb", "default_bob"], ids=["qbb", "bob"], indirect=True)
@pytest.mark.parametrize("policy", ["fnn_policy"], indirect=True)
@pytest.mark.parametrize("algo", [A2C, PPO])
@pytest.mark.parametrize("min_rollouts", [2, 4, 6])  # Once less, equal, and more rollouts than workers.
def test_parallel_sampling_deterministic_smoke_test_wo_min_steps(
    tmpdir_factory, env: SimEnv, policy: Policy, algo, min_rollouts: int
):
    seeds = (0, 1)
    nums_workers = (1, 2, 4)

    logging_results = []
    rollout_results: List[List[List[List[StepSequence]]]] = []
    for seed in seeds:
        logging_results.append((seed, []))
        rollout_results.append([])
        for num_workers in nums_workers:
            pyrado.set_seed(seed)
            policy.init_param(None)
            ex_dir = str(tmpdir_factory.mktemp(f"seed={seed}-num_workers={num_workers}"))
            set_log_prefix_dir(ex_dir)
            vfcn = FNN(input_size=env.obs_space.flat_dim, output_size=1, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
            critic = GAE(vfcn, gamma=0.98, lamda=0.95, batch_size=32, lr=1e-3, standardize_adv=False)
            alg = algo(ex_dir, env, policy, critic, max_iter=3, min_rollouts=min_rollouts, num_workers=num_workers)
            alg.sampler = RolloutSavingWrapper(alg.sampler)
            alg.train()
            with open(f"{ex_dir}/progress.csv") as f:
                logging_results[-1][1].append(str(f.read()))
            rollout_results[-1].append(alg.sampler.rollouts)

    # Test that the observations for all number of workers are equal.
    for rollouts in rollout_results:
        for ros_a, ros_b in [(a, b) for a in rollouts for b in rollouts]:
            assert len(ros_a) == len(ros_b)
            for ro_a, ro_b in zip(ros_a, ros_b):
                assert len(ro_a) == len(ro_b)
                for r_a, r_b in zip(ro_a, ro_b):
                    assert r_a.observations == pytest.approx(r_b.observations)

    # Test that different seeds actually produce different results.
    for results_a, results_b in [
        (a, b) for seed_a, a in logging_results for seed_b, b in logging_results if seed_a != seed_b
    ]:
        for result_a, result_b in [(a, b) for a in results_a for b in results_b if a is not b]:
            assert result_a != result_b

    # Test that same seeds produce same results.
    for _, results in logging_results:
        for result_a, result_b in [(a, b) for a in results for b in results]:
            assert result_a == result_b
