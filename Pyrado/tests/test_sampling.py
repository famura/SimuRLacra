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
from matplotlib import pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal

from pyrado.domain_randomization.default_randomizers import get_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.policies.fnn import FNNPolicy
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
from tests.conftest import m_needs_cuda


@pytest.mark.parametrize(
    'arg', [
        [1],
        [2, 3],
        [4, 6, 2, 88, 3, 45, 7, 21, 22, 23, 24, 44, 45, 56, 67, 78, 89],
    ]
)
def test_sampler_pool(arg):
    pool = SamplerPool(len(arg))
    result = pool.invoke_all_map(_cb_test_eachhandler, arg)
    pool.stop()

    assert result == list(map(lambda x: x*2, arg))


def _cb_test_eachhandler(G, arg):
    time.sleep(random.randint(1, 5))
    return arg*2


def _cb_test_collecthandler(G):
    nsample = random.randint(5, 15)
    return nsample, nsample


@pytest.mark.sampling
@pytest.mark.parametrize(
    'num_threads', [1, 2, 4]
)
@pytest.mark.parametrize(
    'min_samples', [10, 20, 40]
)
def test_sampler_collect(num_threads, min_samples):
    pool = SamplerPool(num_threads)

    # Run the collector
    cr, cn = pool.run_collect(min_samples, _cb_test_collecthandler)
    pool.stop()

    assert min_samples <= cn
    assert min_samples <= sum(cr)


@pytest.mark.sampling
@pytest.mark.parametrize(
    'num_threads', [1, 2, 4]
)
@pytest.mark.parametrize(
    'min_samples', [10, 20, 40]
)
@pytest.mark.parametrize(
    'min_runs', [10, 20, 40]
)
def test_sampler_collect_minrun(num_threads, min_samples, min_runs):
    pool = SamplerPool(num_threads)

    # Run the collector
    cr, cn = pool.run_collect(min_samples, _cb_test_collecthandler, min_runs=min_runs)

    pool.stop()

    assert min_samples <= cn
    assert min_samples <= sum(cr)
    assert min_runs <= len(cr)


@pytest.mark.sampling
@pytest.mark.parametrize(
    'data_type', [
        (None, None), (to.int32, np.int32),
    ]
)
def test_to_format(data_type):
    # Create some tensors to convert
    ndarray = np.random.rand(3, 2).astype(dtype=np.float64)
    tensor = to.rand(3, 2).type(dtype=to.float64)

    # Test the conversion and typing from numpy to PyTorch
    converted_ndarray = to_format(ndarray, 'torch', data_type[0])
    assert isinstance(converted_ndarray, to.Tensor)
    new_type = to.float64 if data_type[0] is None else data_type[0]  # passing None must not change the type
    assert converted_ndarray.dtype == new_type

    # Test the conversion and typing from PyTorch to numpy
    converted_tensor = to_format(tensor, 'numpy', data_type[1])
    assert isinstance(converted_tensor, np.ndarray)
    new_type = np.float64 if data_type[1] is None else data_type[1]  # passing None must not change the type
    assert converted_tensor.dtype == new_type


@pytest.mark.sampling
@pytest.mark.parametrize(
    'epsilon', [
        1, 0.5, 0.1,
    ]
)
@pytest.mark.parametrize(
    'num_ro', [
        10, 20,
    ]
)
def test_select_cvar(epsilon, num_ro):
    # Create rollouts with known discounted rewards
    rollouts = [
        StepSequence(rewards=[i], observations=[i], actions=[i])
        for i in range(num_ro)
    ]
    # Shuffle data to put in
    ro_shuf = list(rollouts)
    random.shuffle(ro_shuf)

    # Select cvar quantile
    ro_cv = select_cvar(ro_shuf, epsilon, 1)

    # Compute expected return of subselection
    cv = sum(map(lambda ro: ro.discounted_return(1), ro_cv))/len(ro_cv)

    # This should be equal to the epsilon-quantile of the integer sequence
    nq = int(num_ro*epsilon)
    cv_expected = sum(range(nq))/nq

    assert cv == cv_expected


@pytest.mark.sampling
@pytest.mark.parametrize(
    'num_dim, method', [
        (1, 'uniform'), (1, 'uniform'),
        (3, 'uniform'), (3, 'normal'), (3, 'Marsaglia'),
        (4, 'uniform'), (4, 'normal'), (4, 'Marsaglia'),
        (15, 'uniform'), (15, 'normal')
    ]
)
def test_sample_from_unit_sphere_surface(num_dim, method):
    s = sample_from_hyper_sphere_surface(num_dim, method)
    assert 0.95 <= to.norm(s, p=2) <= 1.05


@pytest.mark.parametrize(
    ['env', 'policy'], [
        ('default_bob', 'idle_policy'),
        ('default_bob', 'dummy_policy'),
        ('default_bob', 'time_policy'),
        ('default_bob', 'linear_policy'),
        ('default_bob', 'fnn_policy'),
        ('default_bob', 'rnn_policy'),
        ('default_bob', 'lstm_policy'),
        ('default_bob', 'gru_policy'),
        ('default_bob', 'adn_policy'),
        ('default_bob', 'nf_policy'),
        ('default_bob', 'thfnn_policy'),
        ('default_bob', 'thgru_policy'),
    ], ids=['bob_idle', 'bob_dummy', 'bob_time', 'bob_lin', 'bob_fnn', 'bob_rnn', 'bob_lstm', 'bob_gru', 'bob_adn',
            'bob_nf', 'bob_thfnn', 'bob_thgru'],
    indirect=True)
def test_rollout_wo_exploration(env, policy):
    ro = rollout(env, policy, render_mode=RenderMode())
    assert isinstance(ro, StepSequence)
    assert len(ro) <= env.max_steps


@pytest.mark.parametrize('env', ['default_bob', 'default_qbb'], ids=['bob', 'qbb'], indirect=True)
def test_rollout_wo_policy(env):
    def policy(obs):
        # Callable must receive and return tensors
        return to.from_numpy(env.spec.act_space.sample_uniform())

    ro = rollout(env, policy, render_mode=RenderMode())
    assert isinstance(ro, StepSequence)
    assert len(ro) <= env.max_steps


@pytest.mark.parametrize(
    'mean, cov', [
        (to.tensor([5., 7.]), to.tensor([[2., 0.], [0., 2.]])),
    ], ids=['2dim']
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


@pytest.mark.sampling
@pytest.mark.visualization
@pytest.mark.parametrize(
    'sequence, x_init', [
        # (sequence_const, np.array([2])),
        # (sequence_plus_one, np.array([2])),
        # (sequence_add_init, np.array([2])),
        # (sequence_rec_double, np.array([2])),
        # (sequence_rec_sqrt, np.array([2])),
        # (sequence_nlog2, np.array([2])),
        (sequence_const, np.array([1, 2, 3])),
        (sequence_plus_one, np.array([1, 2, 3])),
        (sequence_add_init, np.array([1, 2, 3])),
        (sequence_rec_double, np.array([1, 2, 3])),
        (sequence_rec_sqrt, np.array([1, 2, 3])),
        (sequence_nlog2, np.array([1, 2, 3])),
    ]
)
def test_sequences(sequence, x_init):
    # Get the full sequence
    _, x_full = sequence(x_init, 5, float)

    # Plot the sequences
    for i in range(x_full.shape[1]):
        plt.stem(x_full[:, i], label=str(x_init[i]))
    plt.legend()
    plt.show()


def test_bootsrapping():
    # Why you should operate on the deltas and not directly on the statistic from the resampled data
    sample = np.array([30, 37, 36, 43, 42, 43, 43, 46, 41, 42])
    mean = np.mean(sample)
    print(mean)
    m, ci = bootstrap_ci(sample, np.mean, num_reps=20, alpha=0.1, ci_sides=2, seed=123)
    print(m, ci)

    np.random.seed(123)
    resampled = np.random.choice(sample, (sample.shape[0], 20), replace=True)
    means = np.apply_along_axis(np.mean, 0, resampled)
    print(np.sort(means))
    ci_lo, ci_up = np.percentile(means, [100*0.05, 100*0.95])
    print(ci_lo, ci_up)

    x = np.random.normal(10, 1, 40)
    # x = np.random.uniform(5, 15, 20)
    # x = np.random.poisson(5, 30)
    np.random.seed(1)
    # print(bs.bootstrap(x, stat_func=bs_stats.mean))

    np.random.seed(1)
    m, ci = bootstrap_ci(x, np.mean, num_reps=1000, alpha=0.05, ci_sides=2, studentized=False, bias_correction=False)
    print('[use_t_for_ci=False] mean: ', m)
    print('[use_t_for_ci=False] CI: ', ci)

    np.random.seed(1)
    m, ci = bootstrap_ci(x, np.mean, num_reps=1000, alpha=0.05, ci_sides=2, studentized=False, bias_correction=True)
    print('[bias_correction=True] mean: ', m)

    m, ci = bootstrap_ci(x, np.mean, num_reps=2*384, alpha=0.05, ci_sides=1, studentized=False)
    print('[use_t_for_ci=False] mean: ', m)
    print('[use_t_for_ci=False] CI: ', ci)

    m, ci = bootstrap_ci(x, np.mean, num_reps=2*384, alpha=0.05, ci_sides=1, studentized=True)
    print('[use_t_for_ci=True] mean: ', m)
    print('[use_t_for_ci=True] CI: ', ci)

    print('Matlab example:')
    # https://de.mathworks.com/help/stats/bootci.htmls
    x_matlab = np.random.normal(1, 1, 40)

    m, ci = bootstrap_ci(x_matlab, np.mean, num_reps=2000, alpha=0.05, ci_sides=2, studentized=False)
    print('[use_t_for_ci=False] mean: ', m)
    print('[use_t_for_ci=False] CI: ', ci)

    m, ci = bootstrap_ci(x_matlab, np.mean, num_reps=2000, alpha=0.05, ci_sides=2, studentized=True)
    print('[use_t_for_ci=True] mean: ', m)
    print('[use_t_for_ci=True] CI: ', ci)


@pytest.mark.parametrize(
    ['env', 'policy'], [
        ('default_bob', 'fnn_policy'),
    ], ids=['bob_fnnpol'], indirect=True)
def test_param_expl_sampler(env, policy):
    # Add randomizer
    pert = get_default_randomizer(env)
    env = DomainRandWrapperLive(env, pert)

    # Create the sampler
    num_rollouts_per_param = 12
    sampler = ParameterExplorationSampler(env, policy, num_workers=1, num_rollouts_per_param=num_rollouts_per_param)

    # Use some random parameters
    num_ps = 12
    params = to.rand(num_ps, policy.num_param)

    # Do the sampling
    samples = sampler.sample(params)

    assert num_ps == len(samples)
    for ps in samples:
        assert len(ps.rollouts) == num_rollouts_per_param

    # Compare rollouts that should be matching
    for ri in range(num_rollouts_per_param):
        # Use the first paramset as pivot
        piter = iter(samples)
        pivot = next(piter).rollouts[ri]
        # Iterate through others
        for ops in piter:
            ro = ops.rollouts[ri]

            # Compare domain params
            assert pivot.rollout_info['domain_param'] == ro.rollout_info['domain_param']
            # Compare first observation a.k.a. init state
            assert pivot[0].observation == pytest.approx(ro[0].observation)


@m_needs_cuda
def test_cuda_sampling_w_dr(default_bob, bob_pert):
    # Add randomizer
    env = DomainRandWrapperLive(default_bob, bob_pert)

    # Use a simple policy
    policy = FNNPolicy(env.spec, hidden_sizes=[8], hidden_nonlin=to.tanh, use_cuda=True)

    # Create the sampler
    sampler = ParallelRolloutSampler(env, policy, num_workers=2, min_rollouts=10)

    samples = sampler.sample()
    assert samples is not None
