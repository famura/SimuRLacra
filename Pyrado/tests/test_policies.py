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

import pytest
from tests.conftest import m_needs_bullet, m_needs_cuda, m_needs_libtorch, m_needs_mujoco, m_needs_rcs
from torch import nn as nn

import pyrado
from pyrado.environments.base import Env
from pyrado.policies.base import Policy
from pyrado.policies.features import *
from pyrado.policies.feed_back.dual_rfb import DualRBFLinearPolicy
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.policies.feed_forward.playback import PlaybackPolicy
from pyrado.policies.feed_forward.poly_time import PolySplineTimePolicy
from pyrado.policies.recurrent.base import RecurrentPolicy, default_pack_hidden, default_unpack_hidden
from pyrado.policies.recurrent.two_headed_rnn import TwoHeadedRNNPolicyBase
from pyrado.policies.special.environment_specific import (
    QBallBalancerPDCtrl,
    QCartPoleSwingUpAndBalanceCtrl,
    QQubeSwingUpAndBalanceCtrl,
)
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.data_types import RenderMode
from pyrado.utils.nn_layers import IndiNonlinLayer


@pytest.mark.features
@pytest.mark.parametrize(
    "feat_list",
    [
        [const_feat],
        [identity_feat],
        [
            const_feat,
            identity_feat,
            sign_feat,
            abs_feat,
            squared_feat,
            cubic_feat,
            sig_feat,
            bell_feat,
            sin_feat,
            cos_feat,
            sinsin_feat,
            sincos_feat,
        ],
    ],
    ids=["const_only", "ident_only", "all_simple_feats"],
)
def test_simple_feature_stack(feat_list: list):
    fs = FeatureStack(*feat_list)
    obs = to.randn(1)
    feats_val = fs(obs)
    assert isinstance(feats_val, to.Tensor)


@pytest.mark.features
@pytest.mark.parametrize("obs_dim, idcs", [(2, (0, 1)), (3, (2, 0)), (10, (0, 1, 5, 6))], ids=["2_2", "3_2", "10_4"])
def test_mul_feat(obs_dim: int, idcs: tuple):
    fs = FeatureStack(identity_feat, MultFeat(idcs=idcs))
    obs = to.randn(obs_dim)
    feats_val = fs(obs)
    assert isinstance(feats_val, to.Tensor)
    assert len(feats_val) == obs_dim + 1


@pytest.mark.features
@pytest.mark.parametrize(
    "obs_dim, num_feat_per_dim", [(1, 1), (2, 1), (1, 4), (2, 4), (10, 100)], ids=["1_1", "2_1", "1_4", "2_4", "10_100"]
)
def test_rff_feat_serial(obs_dim: int, num_feat_per_dim: int):
    rff = RFFeat(
        inp_dim=obs_dim,
        num_feat_per_dim=num_feat_per_dim,
        bandwidth=np.ones(obs_dim),
    )
    fs = FeatureStack(rff)
    for _ in range(10):
        obs = to.randn(obs_dim)
        feats_val = fs(obs)
        assert isinstance(feats_val, to.Tensor)
        assert feats_val.shape == (1, num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize("batch_size", [1, 2, 100], ids=["1", "2", "100"])
@pytest.mark.parametrize(
    "obs_dim, num_feat_per_dim", [(1, 1), (2, 1), (1, 4), (2, 4), (10, 100)], ids=["1_1", "2_1", "1_4", "2_4", "10_100"]
)
def test_rff_feat_batched(batch_size: int, obs_dim: int, num_feat_per_dim: int):
    rff = RFFeat(
        inp_dim=obs_dim,
        num_feat_per_dim=num_feat_per_dim,
        bandwidth=np.ones(obs_dim),
    )
    fs = FeatureStack(rff)
    for _ in range(10):
        obs = to.randn(batch_size, obs_dim)
        feats_val = fs(obs)
        assert isinstance(feats_val, to.Tensor)
        assert feats_val.shape == (batch_size, num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    "obs_dim, num_feat_per_dim, bounds",
    [
        (1, 4, (to.tensor([-3.0]), to.tensor([3.0]))),
        (1, 4, (np.array([-3.0]), np.array([3.0]))),
        (2, 4, (to.tensor([-3.0, -4.0]), to.tensor([3.0, 4.0]))),
        (10, 100, (to.tensor([-3.0] * 10), to.tensor([3.0] * 10))),
    ],
    ids=["1_4_to", "1_4_np", "2_4", "10_100"],
)
def test_rbf_serial(obs_dim: int, num_feat_per_dim: int, bounds: to.Tensor):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=bounds)
    fs = FeatureStack(rbf)
    for _ in range(10):
        obs = to.randn(obs_dim)  # 1-dim obs vector
        feats_val = fs(obs)
        assert isinstance(feats_val, to.Tensor)
        assert feats_val.shape == (1, obs_dim * num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize("batch_size", [1, 2, 100], ids=["1", "2", "100"])
@pytest.mark.parametrize(
    "obs_dim, num_feat_per_dim, bounds",
    [
        (1, 4, (to.tensor([-3.0]), to.tensor([3.0]))),
        (1, 4, (np.array([-3.0]), np.array([3.0]))),
        (2, 4, (to.tensor([-3.0, -4.0]), to.tensor([3.0, 4.0]))),
        (10, 100, (to.tensor([-3.0] * 10), to.tensor([3.0] * 10))),
    ],
    ids=["1_4_to", "1_4_np", "2_4", "10_100"],
)
def test_rbf_feat_batched(batch_size: int, obs_dim: int, num_feat_per_dim: int, bounds: to.Tensor):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=bounds)
    fs = FeatureStack(rbf)
    for _ in range(10):
        obs = to.randn(batch_size, obs_dim)  # 2-dim obs array
        feats_val = fs(obs)
        assert isinstance(feats_val, to.Tensor)
        assert feats_val.shape == (batch_size, obs_dim * num_feat_per_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qqsu",
        "default_qbb",
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
    ],
    ids=["bob", "qq-st", "qbb", "bop5D"],
    indirect=True,
)
@pytest.mark.parametrize("num_feat_per_dim", [4, 100], ids=["4", "100"])
def test_rff_policy_serial(env: Env, num_feat_per_dim: int):
    rff = RFFeat(inp_dim=env.obs_space.flat_dim, num_feat_per_dim=num_feat_per_dim, bandwidth=env.obs_space.bound_up)
    policy = LinearPolicy(env.spec, FeatureStack(rff))
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
        act = policy(obs)
        assert act.shape == (env.act_space.flat_dim,)


@pytest.mark.features
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qqsu",
        "default_qbb",
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
    ],
    ids=["bob", "qq-su", "qbb", "bop5D"],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, num_feat_per_dim", [(1, 4), (20, 4), (1, 100), (20, 100)], ids=["1_4", "20_4", "1_100", "20_100"]
)
def test_rff_policy_batch(env: Env, batch_size: int, num_feat_per_dim: int):
    rff = RFFeat(inp_dim=env.obs_space.flat_dim, num_feat_per_dim=num_feat_per_dim, bandwidth=env.obs_space.bound_up)
    policy = LinearPolicy(env.spec, FeatureStack(rff))
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
        obs = obs.repeat(batch_size, 1)
        act = policy(obs)
        assert act.shape == (batch_size, env.act_space.flat_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qqsu",
        "default_qbb",
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
    ],
    ids=["bob", "qq-su", "qbb", "bop5D"],
    indirect=True,
)
@pytest.mark.parametrize("num_feat_per_dim", [4, 100], ids=["4", "100"])
def test_rfb_policy_serial(env: Env, num_feat_per_dim: int):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=env.obs_space.bounds)
    fs = FeatureStack(rbf)
    policy = LinearPolicy(env.spec, fs)
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
        act = policy(obs)
        assert act.shape == (env.act_space.flat_dim,)


@pytest.mark.features
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qqsu",
        "default_qbb",
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
    ],
    ids=["bob", "qq-su", "qbb", "bop5D"],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size, num_feat_per_dim", [(1, 4), (20, 4), (1, 100), (20, 100)], ids=["1_4", "20_4", "1_100", "20_100"]
)
def test_rfb_policy_batch(env: Env, batch_size: int, num_feat_per_dim: int):
    rbf = RBFFeat(num_feat_per_dim=num_feat_per_dim, bounds=env.obs_space.bounds)
    fs = FeatureStack(rbf)
    policy = LinearPolicy(env.spec, fs)
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
        obs = obs.repeat(batch_size, 1)
        act = policy(obs)
        assert act.shape == (batch_size, env.act_space.flat_dim)


@pytest.mark.features
@pytest.mark.parametrize(
    "env",
    [
        pytest.param("default_wambic", marks=m_needs_mujoco),  # so far, the only use case
    ],
    ids=["wambic"],
    indirect=True,
)
@pytest.mark.parametrize("dim_mask", [0, 1, 2], ids=["0", "1", "2"])
def test_dualrbf_policy(env: Env, dim_mask: int):
    # Hyper-parameters for the RBF features are not important here
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.0]), np.array([1.0])), scale=None)
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam, dim_mask)
    assert policy.num_param == policy.num_active_feat * env.act_space.flat_dim // 2

    ro = rollout(env, policy, eval=True)
    assert isinstance(ro, StepSequence)


@pytest.mark.parametrize(
    "env",
    ["default_qbb", "default_qcpsu", "default_qcpst", "default_qqsu", "default_qqst"],
    ids=["qbb", "qcpsu", "qcpst", "qqsu", "qqst"],
    indirect=True,
)
def test_env_specific(env: Env):
    pyrado.set_seed(0)

    if "qbb" in env.name:
        policy = QBallBalancerPDCtrl(env.spec)
        policy.reset()
    elif "qcp" in env.name:
        policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
        policy.reset()
    elif "qq" in env.name:
        policy = QQubeSwingUpAndBalanceCtrl(env.spec)
        policy.reset()
    else:
        raise NotImplementedError

    # Sample an observation and do an action 10 times
    for _ in range(10):
        obs = env.obs_space.sample_uniform()
        obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
        act = policy(obs)
        assert isinstance(act, to.Tensor)


@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
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
def test_parameterized_policies_init_param(env: Env, policy: Policy):
    some_values = to.ones_like(policy.param_values)
    policy.init_param(some_values)
    to.testing.assert_allclose(policy.param_values, some_values)


@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    ["idle_policy", "dummy_policy", "linear_policy", "fnn_policy"],
    ids=["idle", "dummy", "lin", "fnn"],
    indirect=True,
)
def test_feedforward_policy_one_step(env: Env, policy: Policy):
    obs = env.spec.obs_space.sample_uniform()
    obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
    act = policy(obs)
    assert isinstance(act, to.Tensor)


@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    [
        "time_policy",
        "traced_time_policy",
        "pst_policy",
        "traced_pst_policy",
    ],
    ids=["time", "tracedtime", "pst", "tracedpst"],
    indirect=True,
)
def test_time_policy_one_step(env: Env, policy: Policy):
    policy.reset()
    obs = env.obs_space.sample_uniform()
    obs = to.from_numpy(obs)
    act = policy(obs)
    assert isinstance(act, to.Tensor)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    [
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
        "thgru_policy",
    ],
    ids=["rnn", "lstm", "gru", "adn", "nf", "thgru"],
    indirect=True,
)
def test_recurrent_policy_one_step(env: Env, policy: Policy):
    hid = policy.init_hidden()
    obs = env.obs_space.sample_uniform()
    obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
    if isinstance(policy, TwoHeadedRNNPolicyBase):
        act, out2, hid = policy(obs, hid)
        assert isinstance(out2, to.Tensor)
    else:
        act, hid = policy(obs, hid)
    assert isinstance(act, to.Tensor) and isinstance(hid, to.Tensor)


@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qbb",
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
    ],
    ids=["bob", "qbb", "bop5D"],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    [
        # dummy_policy and idle_policy are not supported
        "linear_policy",
        "fnn_policy",
    ],
    ids=["lin", "fnn"],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_feedforward_policy_batching(env: Env, policy: Policy, batch_size: int):
    obs = np.stack([policy.env_spec.obs_space.sample_uniform() for _ in range(batch_size)])  # shape = (batch_size, 4)
    obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())
    act = policy(obs)
    assert act.shape[0] == batch_size


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qbb",
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
    ],
    ids=["bob", "qbb", "bop5D"],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    [
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
        "thrnn_policy",
        "thgru_policy",
        "thlstm_policy",
    ],
    ids=["rnn", "lstm", "gru", "adn", "nf", "thgrnn", "thgru", "thlstm"],
    indirect=True,
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 256])
def test_recurrent_policy_batching(env: Env, policy: Policy, batch_size: int):
    assert policy.is_recurrent
    obs = np.stack([policy.env_spec.obs_space.sample_uniform() for _ in range(batch_size)])  # shape = (batch_size, 4)
    obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())

    # Do this in evaluation mode to disable dropout&co
    policy.eval()

    # Create initial hidden state
    hidden = policy.init_hidden(batch_size)
    # Use a random one to ensure we don't just run into the 0-special-case
    hidden.random_()
    assert hidden.shape == (batch_size, policy.hidden_size)

    if isinstance(policy, TwoHeadedRNNPolicyBase):
        act, _, hid_new = policy(obs, hidden)
    else:
        act, hid_new = policy(obs, hidden)
    assert hid_new.shape == (batch_size, policy.hidden_size)

    if batch_size > 1:
        # Try to use a subset of the batch
        subset = to.arange(batch_size // 2)
        if isinstance(policy, TwoHeadedRNNPolicyBase):
            act_sub, _, hid_sub = policy(obs[subset, :], hidden[subset, :])
        else:
            act_sub, hid_sub = policy(obs[subset, :], hidden[subset, :])
        to.testing.assert_allclose(act_sub, act[subset, :])
        to.testing.assert_allclose(hid_sub, hid_new[subset, :])


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    "env",
    ["default_bob", "default_qbb", pytest.param("default_bop5d_bt", marks=m_needs_bullet)],
    ids=["bob", "qbb", "bop5d"],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    [
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
        "thrnn_policy",
        "thgru_policy",
        "thlstm_policy",
    ],
    ids=["rnn", "lstm", "gru", "adn", "nf", "thgrnn", "thgru", "thlstm"],
    indirect=True,
)
def test_pytorch_recurrent_policy_rollout(env: Env, policy: Policy):
    ro = rollout(env, policy, render_mode=RenderMode())
    assert isinstance(ro, StepSequence)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    "env",
    ["default_bob", "default_qbb", pytest.param("default_bop5d_bt", marks=m_needs_bullet)],
    ids=["bob", "qbb", "bop5d"],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    [
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
        "thrnn_policy",
        "thgru_policy",
        "thlstm_policy",
    ],
    ids=["rnn", "lstm", "gru", "adn", "nf", "thgrnn", "thgru", "thlstm"],
    indirect=True,
)
def test_recurrent_policy_one_step(env: Env, policy: Policy):
    assert policy.is_recurrent
    obs = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(obs).to(dtype=to.get_default_dtype())

    # Do this in evaluation mode to disable dropout & co
    policy.eval()

    # Create initial hidden state
    hidden = policy.init_hidden()
    # Use a random one to ensure we don't just run into the 0-special-case
    hidden = to.rand_like(hidden)
    assert len(hidden) == policy.hidden_size

    # Test general conformity
    if isinstance(policy, TwoHeadedRNNPolicyBase):
        act, otherhead, hid_new = policy(obs, hidden)
        assert len(hid_new) == policy.hidden_size
    else:
        act, hid_new = policy(obs, hidden)
        assert len(hid_new) == policy.hidden_size

    # Test reproducibility
    if isinstance(policy, TwoHeadedRNNPolicyBase):
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
    "env",
    ["default_pend", "default_qbb"],
    ids=["pend", "qbb"],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    ["rnn_policy", "lstm_policy", "gru_policy"],
    ids=["rnn", "lstm", "gru"],
    indirect=True,
)
def test_basic_policy_evaluate_packed_padded_sequences(env: Env, policy: RecurrentPolicy):
    # Test packed padded sequence implementation against old implementation
    def old_evaluate(rollout: StepSequence, hidden_states_name: str = "hidden_states") -> to.Tensor:
        # Set policy, i.e. PyTorch nn.Module, to evaluation mode
        policy.eval()

        # The passed sample collection might contain multiple rollouts.
        act_list = []
        for ro in rollout.iterate_rollouts():
            if hidden_states_name in rollout.data_names:
                # Get initial hidden state from first step
                hidden = policy._unpack_hidden(ro[0][hidden_states_name])
            else:
                # Let the network pick the default hidden state
                hidden = None

            # Reshape observations to match PyTorch's RNN sequence protocol
            obs = ro.get_data_values("observations", True).unsqueeze(1)
            obs = obs.to(device=policy.device, dtype=to.get_default_dtype())

            # Pass the input through hidden RNN layers
            out, _ = policy.rnn_layers(obs, hidden)

            # And through the output layer
            act = policy.output_layer(out.squeeze(1))
            if policy.output_nonlin is not None:
                act = policy.output_nonlin(act)

            # Collect the actions
            act_list.append(act)

        # Set policy, i.e. PyTorch nn.Module, back to training mode
        policy.train()

        return to.cat(act_list)

    # Get some rollouts
    ros = []
    for i in range(5):
        ro = rollout(env, policy, eval=True, render_mode=RenderMode())
        ro.torch(to.get_default_dtype())
        ros.append(ro)

    # Perform concatenation
    cat = StepSequence.concat(ros)

    # Evaluate old and new approaches
    act_old = old_evaluate(cat)
    act_new = policy.evaluate(cat)

    to.testing.assert_allclose(act_old, act_new)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    "env",
    ["default_pend", "default_qbb"],
    ids=["pend", "qbb"],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    ["thrnn_policy", "thgru_policy", "thlstm_policy"],
    ids=["thrnn", "thgru", "thlstm"],
    indirect=True,
)
def test_twoheaded_policy_evaluate_packed_padded_sequences(env: Env, policy: RecurrentPolicy):
    # Test packed padded sequence implementation for custom recurrent neural networks

    # Get some rollouts
    ros = []
    for i in range(5):
        ro = rollout(env, policy, eval=True, render_mode=RenderMode())
        ro.torch(to.get_default_dtype())
        ros.append(ro)

    # Perform concatenation
    cat = StepSequence.concat(ros)

    # Evaluate old and new approaches
    act_new = policy.evaluate(cat)
    assert act_new is not None


@pytest.mark.recurrent_policy
@pytest.mark.parametrize(
    "env",
    ["default_pend", "default_qbb"],
    ids=["pend", "qbb"],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    ["adn_policy", "nf_policy"],
    ids=["adn", "nf"],
    indirect=True,
)
def test_potential_policy_evaluate_packed_padded_sequences(env: Env, policy: RecurrentPolicy):
    # Test packed padded sequence implementation for custom recurrent neural networks

    # Get some rollouts
    ros = []
    for i in range(5):
        ro = rollout(env, policy, eval=True, render_mode=RenderMode())
        ro.torch(to.get_default_dtype())
        ros.append(ro)

    # Perform concatenation
    cat = StepSequence.concat(ros)

    # Evaluate old and new approaches
    act_new = policy.evaluate(cat)
    assert act_new is not None


@pytest.mark.recurrent_policy
def test_hidden_state_packing_batch():
    num_layers = 2
    hidden_size = 2
    batch_size = 2

    unpacked = to.tensor([[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]])  # l1, b1  # l1, b2  # l2, b1  # l2, b2
    packed = to.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

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

    unpacked = to.tensor([[[1.0, 2.0]], [[3.0, 4.0]]])  # l1  # l2
    packed = to.tensor([1.0, 2.0, 3.0, 4.0])

    # Test unpack
    pu = default_unpack_hidden(packed, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(pu, unpacked)

    # Test pack
    up = default_pack_hidden(unpacked, num_layers, hidden_size, batch_size)
    to.testing.assert_allclose(up, packed)


@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    [
        # TimePolicy and Two-headed policies are not supported
        "linear_policy",
        "fnn_policy",
    ],
    ids=["lin", "fnn"],
    indirect=True,
)
def test_script_nonrecurrent(env: Env, policy: Policy):
    # Generate scripted version
    scripted = policy.double().script()

    # Compare results
    sample = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(sample)
    act_reg = policy(obs)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)


@pytest.mark.recurrent_policy
@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    [
        # Two-headed policies are not supported
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
    ],
    ids=["rnn", "lstm", "gru", "adn", "nf"],
    indirect=True,
)
def test_script_recurrent(env: Env, policy: Policy):
    # Generate scripted version
    scripted = policy.double().script()

    # Compare results, tracing hidden manually
    hidden = policy.init_hidden()

    # Run one step
    sample = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(sample)
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)
    # Run second step
    sample = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(sample)
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)

    # Test after reset
    hidden = policy.init_hidden()
    scripted.reset()

    sample = policy.env_spec.obs_space.sample_uniform()
    obs = to.from_numpy(sample)
    act_reg, hidden = policy(obs, hidden)
    act_script = scripted(obs)
    to.testing.assert_allclose(act_reg, act_script)


@to.no_grad()
@m_needs_libtorch
@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    [
        # TimePolicy and Two-headed policies are not supported
        "linear_policy",
        "fnn_policy",
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
    ],
    ids=["lin", "fnn", "rnn", "lstm", "gru", "adn", "nf"],
    indirect=True,
)
@pytest.mark.parametrize("file_type", [".pt", ".zip"], ids=["pt", "zip"])
def test_export_cpp(env: Env, policy: Policy, tmpdir: str, file_type):
    # Generate scripted version (in double mode for CPP compatibility)
    scripted = policy.double().script()

    # Export
    export_file = osp.join(tmpdir, "policy" + file_type)
    scripted.save(export_file)

    # Import again
    loaded = to.jit.load(export_file)

    # Compare a couple of inputs
    for i in range(50):
        obs = policy.env_spec.obs_space.sample_uniform()
        obs_to = to.from_numpy(obs)  # is already double
        act_scripted = scripted(obs_to).cpu().numpy()
        act_loaded = loaded(to.from_numpy(obs)).cpu().numpy()
        assert act_loaded == pytest.approx(act_scripted), f"Wrong action values on step #{i}"

    # Test after reset
    if hasattr(scripted, "reset"):
        scripted.reset()
        loaded.reset()
        assert loaded.hidden.numpy() == pytest.approx(scripted.hidden.numpy()), "Wrong hidden state after reset"

        obs = policy.env_spec.obs_space.sample_uniform()
        obs_to = to.from_numpy(obs)  # is already double
        act_scripted = scripted(obs_to).cpu().numpy()
        act_loaded = loaded(to.from_numpy(obs)).cpu().numpy()
        assert act_loaded == pytest.approx(act_scripted), "Wrong action values after reset"


@to.no_grad()
@m_needs_rcs
@m_needs_libtorch
@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    [
        # TimePolicy and Two-headed policies are not supported
        "linear_policy",
        "fnn_policy",
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
    ],
    ids=["lin", "fnn", "rnn", "lstm", "gru", "adn", "nf"],
    indirect=True,
)
def test_export_rcspysim(env: Env, policy: Policy, tmpdir: str):
    from rcsenv import ControlPolicy

    # Generate scripted version (double mode for CPP compatibility)
    scripted = policy.double().script()
    print(scripted.graph)

    # Export
    export_file = osp.join(tmpdir, "policy.pt")
    to.jit.save(scripted, export_file)

    # Import in C
    cpp = ControlPolicy("torch", export_file)

    # Compare a couple of inputs
    for _ in range(50):
        obs = policy.env_spec.obs_space.sample_uniform()
        obs = to.from_numpy(obs).to(dtype=to.double)
        act_script = scripted(obs).cpu().numpy()
        act_cpp = cpp(obs, policy.env_spec.act_space.flat_dim)
        assert act_cpp == pytest.approx(act_script)

    # Test after reset
    if hasattr(scripted, "reset"):
        scripted.reset()
        cpp.reset()
        obs = policy.env_spec.obs_space.sample_uniform()
        obs = to.from_numpy(obs).to(dtype=to.double)
        act_script = scripted(obs).cpu().numpy()
        act_cpp = cpp(obs, policy.env_spec.act_space.flat_dim)
        assert act_cpp == pytest.approx(act_script)


@pytest.mark.parametrize("in_features", [1, 3], ids=["1dim", "3dim"])
@pytest.mark.parametrize("same_nonlin", [True, False], ids=["same_nonlin", "different_nonlin"])
@pytest.mark.parametrize("bias", [True, False], ids=["bias", "no_bias"])
@pytest.mark.parametrize("weight", [True, False], ids=["weight", "no_weight"])
def test_indi_nonlin_layer(in_features, same_nonlin, bias, weight):
    if not same_nonlin and in_features > 1:
        nonlin = in_features * [to.tanh]
    else:
        nonlin = to.sigmoid
    layer = IndiNonlinLayer(in_features, nonlin, bias, weight)
    assert isinstance(layer, nn.Module)

    i = to.randn(in_features)
    o = layer(i)
    assert isinstance(o, to.Tensor)
    assert i.shape == o.shape


@to.no_grad()
@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize("dtype", ["torch", "numpy"], ids=["torch", "numpy"])
def test_playback_policy(env: Env, dtype):
    # Create 2 recordings of different length
    if dtype == "torch":
        actions = [to.randn(10, env.spec.act_space.flat_dim), to.randn(7, env.spec.act_space.flat_dim)]
    else:
        actions = [np.random.randn(10, env.spec.act_space.flat_dim), np.random.randn(7, env.spec.act_space.flat_dim)]
    policy = PlaybackPolicy(env.spec, act_recordings=actions)

    if dtype == "torch":
        actions = [a.numpy() for a in actions]

    # Sample one rollout and check the actions
    ro = rollout(env, policy)
    assert policy.curr_rec == 0
    assert np.allclose(ro.actions[:10, :], actions[0])
    assert np.allclose(ro.actions[10:, :], np.zeros(env.spec.act_space.flat_dim))

    # Sample another rollout and check the actions
    ro2 = rollout(env, policy)
    assert policy.curr_rec == 1
    assert np.allclose(ro2.actions[:7, :], actions[1])
    assert np.allclose(ro2.actions[7:, :], np.zeros(env.spec.act_space.flat_dim))

    # Check the properties
    policy.curr_step = 3
    assert policy.curr_step == 3
    policy.curr_rec = 0
    assert policy.curr_rec == 0

    policy.reset_curr_rec()
    assert policy.curr_rec == -1


@pytest.mark.parametrize("env", ["default_pend", "default_qbb"], ids=["pend", "qbb"], indirect=True)
@pytest.mark.parametrize("cond_lvl", ["vel", "acc"], ids=["vel", "acc"])
@pytest.mark.parametrize("cond_final", ["zero", "one"], ids=["zero", "one"])
@pytest.mark.parametrize("cond_init", [None, "rand"], ids=["default", "rand"])
@pytest.mark.parametrize("overtime_behavior", ["hold", "zero"], ids=["hold", "zero"])
@pytest.mark.parametrize("use_cuda", [False, pytest.param(True, marks=m_needs_cuda)], ids=["cpu", "cuda"])
def test_poly_time_policy(env: Env, cond_lvl: str, cond_final: str, cond_init, overtime_behavior: str, use_cuda: bool):
    order = 3 if cond_lvl == "vel" else 5
    num_cond = (order + 1) // 2

    if cond_final == "zero":
        cond_final = to.zeros(num_cond, env.act_space.flat_dim)
    elif cond_final == "one":
        cond_final = to.zeros(num_cond, env.act_space.flat_dim)
        cond_final[::num_cond] = 1.0

    if cond_init == "rand":
        cond_init = to.randn(num_cond, env.act_space.flat_dim)

    # Create instance
    policy = PolySplineTimePolicy(
        spec=env.spec,
        dt=env.dt,
        t_end=int(env.max_steps * env.dt),
        cond_lvl=cond_lvl,
        cond_final=cond_final,
        cond_init=cond_init,
        overtime_behavior=overtime_behavior,
        use_cuda=use_cuda,
    )
    policy.reset()

    act_hist = []
    for _ in range(env.max_steps):
        act = policy(None)
        act_hist.append(act.detach().cpu())

        if cond_final == "zero":
            assert act == pytest.approx(to.zeros_like(act))

    if cond_final == "one":
        assert to.allclose(act_hist[-1], to.ones_like(act))

    # Check overtime behavior
    policy.reset()
    act_hist_ot = []
    for _ in range(2 * env.max_steps):
        act = policy(env.obs_space.sample_uniform())
        act_hist_ot.append(act.detach().cpu())

    if overtime_behavior == "hold":
        assert to.allclose(act_hist_ot[-1], act_hist_ot[-1])
    elif overtime_behavior == "zero":
        assert to.allclose(act_hist_ot[-1], to.zeros_like(act))
