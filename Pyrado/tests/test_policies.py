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
import os.path as osp
from torch import nn as nn

from pyrado.algorithms.timeseries_prediction import TSPred
from pyrado.policies.environment_specific import DualRBFLinearPolicy
from pyrado.spaces import BoxSpace
from pyrado.spaces.box import InfBoxSpace
from pyrado.utils.data_sets import TimeSeriesDataSet
from pyrado.utils.functions import skyline
from pyrado.utils.nn_layers import IndiNonlinLayer
from pyrado.policies.rnn import default_unpack_hidden, default_pack_hidden
from pyrado.policies.linear import LinearPolicy
from pyrado.policies.features import *
from pyrado.policies.two_headed import TwoHeadedGRUPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode
from tests.conftest import m_needs_cuda, m_needs_bullet, m_needs_mujoco, m_needs_rcs, m_needs_libtorch
from tests.environment_wrappers.mock_env import MockEnv


@pytest.fixture(scope='function',
                params=[
                    (skyline(0.02, 20., BoxSpace(0.5, 3, shape=(1,)), BoxSpace(-2., 3., shape=(1,)))[1],
                     0.7, 50, False, True),
                ],
                ids=['skyline_0.8_50_notstdized_scaled'])
def tsdataset(request):
    return TimeSeriesDataSet(
        data=to.from_numpy(request.param[0]),
        ratio_train=request.param[1],
        window_size=request.param[2],
        standardize_data=request.param[3],
        scale_min_max_data=request.param[4]
    )


@pytest.mark.features
@pytest.mark.parametrize(
    'feat_list', [
        [const_feat],
        [identity_feat],
        [const_feat, identity_feat, abs_feat, sign_feat, squared_feat, sin_feat, cos_feat, sinsin_feat, sincos_feat,
         sig_feat, bell_feat],
    ], ids=['const_only', 'ident_only', 'all_simple_feats']
)
def test_simple_feature_stack(feat_list):
    fs = FeatureStack(feat_list)
    obs = to.randn(1, )
    feats_val = fs(obs)
    assert feats_val is not None


@pytest.mark.features
@pytest.mark.parametrize(
    'obs_dim, idcs', [
        (2, [0, 1]),
        (3, [2, 0]),
        (10, [0, 1, 5, 6])
    ], ids=['2_2', '3_2', '10_4']
)
def test_mul_feat(obs_dim, idcs):
    mf = MultFeat(idcs=idcs)
    fs = FeatureStack([identity_feat, mf])
    obs = to.randn(obs_dim, )
    feats_val = fs(obs)
    assert len(feats_val) == obs_dim + 1


@pytest.mark.features
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim', [
        (1, 1), (2, 1), (1, 4), (2, 4), (10, 100)
    ], ids=['1_1', '2_1', '1_4', '2_4', '10_100']
)
def test_rff_feat_serial(obs_dim, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=obs_dim, num_feat_per_dim=num_feat_per_dim, bandwidth=np.ones(obs_dim, ))
    fs = FeatureStack([rff])
    for _ in range(10):
        obs = to.randn(obs_dim, )
        feats_val = fs(obs)
        assert feats_val.shape == (1, num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize('batch_size', [1, 2, 100], ids=['1', '2', '100'])
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim', [
        (1, 1), (2, 1), (1, 4), (2, 4), (10, 100)
    ], ids=['1_1', '2_1', '1_4', '2_4', '10_100']
)
def test_rff_feat_batched(batch_size, obs_dim, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=obs_dim, num_feat_per_dim=num_feat_per_dim, bandwidth=np.ones(obs_dim, ))
    fs = FeatureStack([rff])
    for _ in range(10):
        obs = to.randn(batch_size, obs_dim)
        feats_val = fs(obs)
        assert feats_val.shape == (batch_size, num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim, bounds', [
        (1, 4, (to.tensor([-3.]), to.tensor([3.]))),
        (1, 4, (np.array([-3.]), np.array([3.]))),
        (2, 4, (to.tensor([-3., -4.]), to.tensor([3., 4.]))),
        (10, 100, (to.tensor([-3.]*10), to.tensor([3.]*10)))
    ], ids=['1_4_to', '1_4_np', '2_4', '10_100']
)
def test_rbf_serial(obs_dim, num_feat_per_dim, bounds):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=bounds)
    fs = FeatureStack([rbf])
    for _ in range(10):
        obs = to.randn(obs_dim, )  # 1-dim obs vector
        feats_val = fs(obs)
        assert feats_val.shape == (1, obs_dim*num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize('batch_size', [1, 2, 100], ids=['1', '2', '100'])
@pytest.mark.parametrize(
    'obs_dim, num_feat_per_dim, bounds', [
        (1, 4, (to.tensor([-3.]), to.tensor([3.]))),
        (1, 4, (np.array([-3.]), np.array([3.]))),
        (2, 4, (to.tensor([-3., -4.]), to.tensor([3., 4.]))),
        (10, 100, (to.tensor([-3.]*10), to.tensor([3.]*10)))
    ], ids=['1_4_to', '1_4_np', '2_4', '10_100']
)
def test_rbf_feat_batched(batch_size, obs_dim, num_feat_per_dim, bounds):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=bounds)
    fs = FeatureStack([rbf])
    for _ in range(10):
        obs = to.randn(batch_size, obs_dim)  # 2-dim obs array
        feats_val = fs(obs)
        assert feats_val.shape == (batch_size, obs_dim*num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qqsu',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qq-st', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'num_feat_per_dim', [4, 100], ids=['4', '100']
)
def test_rff_policy_serial(env, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=env.obs_space.flat_dim, num_feat_per_dim=num_feat_per_dim,
                          bandwidth=env.obs_space.bound_up)
    policy = LinearPolicy(env.spec, FeatureStack([rff]))
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        act = policy(to.from_numpy(obs))
        assert act.shape == (env.act_space.flat_dim,)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qqsu',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qq-su', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'batch_size, num_feat_per_dim', [
        (1, 4), (20, 4), (1, 100), (20, 100)
    ], ids=['1_4', '20_4', '1_100', '20_100']
)
def test_rff_policy_batch(env, batch_size, num_feat_per_dim):
    rff = RandFourierFeat(inp_dim=env.obs_space.flat_dim, num_feat_per_dim=num_feat_per_dim,
                          bandwidth=env.obs_space.bound_up)
    policy = LinearPolicy(env.spec, FeatureStack([rff]))
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).repeat(batch_size, 1)
        act = policy(obs)
        assert act.shape == (batch_size, env.act_space.flat_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qqsu',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qq-su', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'num_feat_per_dim', [4, 100], ids=['4', '100']
)
def test_rfb_policy_serial(env, num_feat_per_dim):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=env.obs_space.bounds)
    fs = FeatureStack([rbf])
    policy = LinearPolicy(env.spec, fs)
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        act = policy(to.from_numpy(obs))
        assert act.shape == (env.act_space.flat_dim,)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qqsu',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qq-su', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'batch_size, num_feat_per_dim', [
        (1, 4), (20, 4), (1, 100), (20, 100)
    ], ids=['1_4', '20_4', '1_100', '20_100']
)
def test_rfb_policy_batch(env, batch_size, num_feat_per_dim):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=env.obs_space.bounds)
    fs = FeatureStack([rbf])
    policy = LinearPolicy(env.spec, fs)
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).repeat(batch_size, 1)
        act = policy(obs)
        assert act.shape == (batch_size, env.act_space.flat_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    'env', [
        pytest.param('default_wambic', marks=m_needs_mujoco),  # so far, the only use case
    ], ids=['wambic'],
    indirect=True
)
@pytest.mark.parametrize('dim_mask', [0, 1, 2], ids=['0', '1', '2'])
def test_dualrbf_policy(env, dim_mask):
    # Hyper-parameters for the RBF features are not important here
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.]), np.array([1.])), scale=None)
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam, dim_mask)
    assert policy.num_param == policy.num_active_feat*env.act_space.flat_dim//2

    ro = rollout(env, policy, eval=True)
    assert ro is not None


@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb'
    ], ids=['bob', 'qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'linear_policy',
        'fnn_policy',
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
        'thfnn_policy',
        'thgru_policy',
    ],
    ids=['lin', 'fnn', 'rnn', 'lstm', 'gru', 'adn', 'nf', 'thfnn', 'thgru'],
    indirect=True
)
def test_parameterized_policies_init_param(env, policy):
    some_values = to.ones_like(policy.param_values)
    policy.init_param(some_values)
    to.testing.assert_allclose(policy.param_values, some_values)


@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb'
    ], ids=['bob', 'qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'idle_policy',
        'dummy_policy',
        'linear_policy',
        'fnn_policy'
    ], ids=['idle', 'dummy', 'lin', 'fnn'],
    indirect=True)
def test_feedforward_policy_one_step(env, policy):
    obs = env.spec.obs_space.sample_uniform()
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert isinstance(act, to.Tensor)


@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb'
    ], ids=['bob', 'qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'time_policy',
        'tracetime_policy',
    ],
    ids=['time', 'tracetime'],
    indirect=True
)
def test_time_policy_one_step(env, policy):
    policy.reset()
    obs = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert isinstance(act, to.Tensor)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb'
    ], ids=['bob', 'qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
        'thgru_policy',
    ],
    ids=['rnn', 'lstm', 'gru', 'adn', 'nf', 'thgru'],
    indirect=True
)
def test_recurrent_policy_one_step(env, policy):
    hid = policy.init_hidden()
    obs = env.obs_space.sample_uniform()
    obs = to.from_numpy(obs)
    if isinstance(policy, TwoHeadedGRUPolicy):
        act, out2, hid = policy(obs, hid)
        assert isinstance(out2, to.Tensor)
    else:
        act, hid = policy(obs, hid)
    assert isinstance(act, to.Tensor) and isinstance(hid, to.Tensor)


@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        # dummy_policy and idle_policy are not supported
        'linear_policy',
        'fnn_policy',
    ], ids=['lin', 'fnn'], indirect=True)
@pytest.mark.parametrize('batch_size', [1, 2, 3])
def test_feedforward_policy_batching(env, policy, batch_size):
    obs = np.stack([policy.env_spec.obs_space.sample_uniform() for _ in range(batch_size)])  # shape = (batch_size, 4)
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert act.shape[0] == batch_size


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
    ], ids=['rnn', 'lstm', 'gru', 'adn'],
    indirect=True
)
@pytest.mark.parametrize('batch_size', [1, 2, 4])
def test_recurrent_policy_batching(env, policy, batch_size):
    assert policy.is_recurrent
    obs = np.stack([policy.env_spec.obs_space.sample_uniform() for _ in range(batch_size)])  # shape = (batch_size, 4)
    obs = to.from_numpy(obs)

    # Do this in evaluation mode to disable dropout&co
    policy.eval()

    # Create initial hidden state
    hidden = policy.init_hidden(batch_size)
    # Use a random one to ensure we don't just run into the 0-special-case
    hidden.random_()
    assert hidden.shape == (batch_size, policy.hidden_size)

    act, hid_new = policy(obs, hidden)
    assert hid_new.shape == (batch_size, policy.hidden_size)

    if batch_size > 1:
        # Try to use a subset of the batch
        subset = to.arange(batch_size//2)
        act_sub, hid_sub = policy(obs[subset, :], hidden[subset, :])

        to.testing.assert_allclose(act_sub, act[subset, :])
        to.testing.assert_allclose(hid_sub, hid_new[subset, :])


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb'
    ], ids=['bob', 'qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'rnn_policy',
        'lstm_policy',
        'gru_policy'
    ], ids=['rnn', 'lstm', 'gru'], indirect=True)
def test_pytorch_recurrent_policy_rollout(env, policy):
    ro = rollout(env, policy, render_mode=RenderMode())
    assert isinstance(ro, StepSequence)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb'
    ], ids=['bob', 'qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
        'thgru_policy',
    ], ids=['rnn', 'lstm', 'gru', 'adn', 'nf', 'thgru'], indirect=True)
def test_recurrent_policy_one_step(env, policy):
    assert policy.is_recurrent
    obs = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(obs)

    # Do this in evaluation mode to disable dropout & co
    policy.eval()

    # Create initial hidden state
    hidden = policy.init_hidden()
    # Use a random one to ensure we don't just run into the 0-special-case
    hidden = to.rand_like(hidden)
    assert len(hidden) == policy.hidden_size

    # Test general conformity
    if isinstance(policy, TwoHeadedGRUPolicy):
        act, otherhead, hid_new = policy(obs, hidden)
        assert len(hid_new) == policy.hidden_size
    else:
        act, hid_new = policy(obs, hidden)
        assert len(hid_new) == policy.hidden_size

    # Test reproducibility
    if isinstance(policy, TwoHeadedGRUPolicy):
        act2, otherhead2, hid_new2 = policy(obs, hidden)
        to.testing.assert_allclose(act, act2)
        to.testing.assert_allclose(otherhead, otherhead2)
        to.testing.assert_allclose(hid_new2, hid_new2)
    else:
        act2, hid_new2 = policy(obs, hidden)
        to.testing.assert_allclose(act, act2)
        to.testing.assert_allclose(hid_new2, hid_new2)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
    ], ids=['rnn', 'lstm', 'gru', 'adn', 'nf'],
    indirect=True
)
def test_recurrent_policy_evaluate(env, policy):
    # Make a rollout
    ro = rollout(env, policy, eval=True, render_mode=RenderMode())
    ro.torch(to.get_default_dtype())

    # Evaluate first and second action manually
    o1 = ro[0].observation
    h1 = ro[0].hidden_state
    a1, h2 = policy(o1, h1)
    to.testing.assert_allclose(a1.detach(), ro[0].action)
    to.testing.assert_allclose(h2.detach(), ro[0].next_hidden_state)

    # Run evaluate
    eval_act = policy.evaluate(ro)

    to.testing.assert_allclose(eval_act.detach(), ro.actions)


@pytest.mark.recurrent_policy
def test_hidden_state_packing_batch():
    num_layers = 2
    hidden_size = 2
    batch_size = 2

    unpacked = to.tensor([[[1.0, 2.0],  # l1, b1
                           [5.0, 6.0]],  # l1, b2
                          [[3.0, 4.0],  # l2, b1
                           [7.0, 8.0]]])  # l2, b2
    packed = to.tensor([[1.0, 2.0, 3.0, 4.0],
                        [5.0, 6.0, 7.0, 8.0]])

    # Test unpack
    pu = default_unpack_hidden(packed, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(pu, unpacked)

    # Test pack
    up = default_pack_hidden(unpacked, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(up, packed)


@pytest.mark.recurrent_policy
def test_hidden_state_packing_nobatch():
    num_layers = 2
    hidden_size = 2
    batch_size = None

    unpacked = to.tensor([[[1.0, 2.0]],  # l1
                          [[3.0, 4.0]]])  # l2
    packed = to.tensor([1.0, 2.0, 3.0, 4.0])

    # Test unpack
    pu = default_unpack_hidden(packed, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(pu, unpacked)

    # Test pack
    up = default_pack_hidden(unpacked, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(up, packed)


@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        # TimePolicy and Two-headed policies are not supported
        'linear_policy',
        'fnn_policy',
    ], ids=['lin', 'fnn'],
    indirect=True
)
def test_script_feedforward(env, policy):
    # Generate scripted version
    scripted = policy.script()

    # Compare results
    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())

    act_reg = policy(obs)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
    ], ids=['rnn', 'lstm', 'gru', 'adn', 'nf'],
    indirect=True
)
def test_script_recurrent(env, policy):
    # Generate scripted version
    scripted = policy.script()

    # Compare results, tracing hidden manually
    hidden = policy.init_hidden()

    # Run one step
    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)
    # Run second step
    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)

    # Test after reset
    hidden = policy.init_hidden()
    scripted.reset()

    obs = to.from_numpy(policy.env_spec.obs_space.sample_uniform())
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)


@to.no_grad()
@m_needs_libtorch
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        # TimePolicy and Two-headed policies are not supported
        'linear_policy',
        'fnn_policy',
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
    ], ids=['lin', 'fnn', 'rnn', 'lstm', 'gru', 'adn', 'nf'],
    indirect=True
)
@pytest.mark.parametrize(
    'file_type', ['.pt', '.zip'], ids=['pt', 'zip']
)
def test_export_cpp(env, policy, tmpdir, file_type):
    # Generate scripted version (in double mode for CPP compatibility)
    scripted = policy.double().script()

    # Export
    export_file = osp.join(tmpdir, 'policy' + file_type)
    scripted.save(export_file)

    # Import again
    loaded = to.jit.load(export_file)

    # Compare a couple of inputs
    for _ in range(50):
        obs = policy.env_spec.obs_space.sample_uniform()
        act_scripted = scripted(to.from_numpy(obs)).cpu().numpy()
        act_loaded = loaded(to.from_numpy(obs)).cpu().numpy()
        assert act_loaded == pytest.approx(act_scripted)

    # Test after reset
    if hasattr(scripted, 'reset'):
        scripted.reset()
        loaded.reset()

        obs = policy.env_spec.obs_space.sample_uniform()
        act_scripted = scripted(to.from_numpy(obs)).numpy()
        act_loaded = loaded(to.from_numpy(obs)).numpy()
        assert act_loaded == pytest.approx(act_scripted)


@to.no_grad()
@m_needs_rcs
@m_needs_libtorch
@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb',
        pytest.param('default_bop5d_bt', marks=m_needs_bullet),
    ], ids=['bob', 'qbb', 'bop5D'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        # TimePolicy and Two-headed policies are not supported
        'linear_policy',
        'fnn_policy',
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
    ], ids=['lin', 'fnn', 'rnn', 'lstm', 'gru', 'adn', 'nf'],
    indirect=True
)
def test_export_rcspysim(env, policy, tmpdir):
    from rcsenv import ControlPolicy

    # Generate scripted version (in double mode for CPP compatibility)
    scripted = policy.double().script()
    print(scripted.graph)

    # Export
    export_file = osp.join(tmpdir, 'policy.pt')
    to.jit.save(scripted, export_file)

    # Import in C
    cpp = ControlPolicy('torch', export_file)

    # Compare a couple of inputs
    for _ in range(50):
        obs = policy.env_spec.obs_space.sample_uniform()
        act_script = scripted(to.from_numpy(obs)).numpy()
        act_cpp = cpp(obs, policy.env_spec.act_space.flat_dim)
        assert act_cpp == pytest.approx(act_script)

    # Test after reset
    if hasattr(scripted, 'reset'):
        scripted.reset()
        cpp.reset()
        obs = policy.env_spec.obs_space.sample_uniform()
        act_script = scripted(to.from_numpy(obs)).numpy()
        act_cpp = cpp(obs, policy.env_spec.act_space.flat_dim)
        assert act_cpp == pytest.approx(act_script)


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


@pytest.mark.parametrize(
    'env', [MockEnv(obs_space=InfBoxSpace(shape=1), act_space=InfBoxSpace(shape=1))]
)
@pytest.mark.parametrize(
    'policy', [
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'nf_policy',
    ], ids=['rnn', 'lstm', 'gru', 'adn', 'nf'],
    indirect=True
)
@pytest.mark.parametrize('windowed', [True, False], ids=['windowed', 'not_windowed'])
@pytest.mark.parametrize('cascaded', [True, False], ids=['cascaded', 'not_cascaded'])
def test_tspred(tsdataset, env, policy, windowed, cascaded):
    if windowed:
        inp_seq = tsdataset.data_trn_inp
    else:
        inp_seq = tsdataset.data_trn_ws

    # Make the predictions
    preds, hidden = TSPred.predict(policy, inp_seq, windowed, cascaded, hidden=None)

    # Check types
    assert isinstance(preds, to.Tensor)
    assert isinstance(hidden, to.Tensor)
    # Check sizes
    if windowed:
        assert preds.shape[0] == 1
    else:
        assert preds.shape[0] == inp_seq.shape[0]
    assert preds.shape[1] == env.spec.act_space.flat_dim
    assert hidden.numel() == policy.hidden_size
