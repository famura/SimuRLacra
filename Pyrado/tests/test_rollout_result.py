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

"""
NOTE: This file is not using to.testing.assert_allclose because most methods need to work for both torch and numpy.
"""
import pytest
import numpy as np
import torch as to
import itertools
import pickle
from pytest_lazyfixture import lazy_fixture
from typing import NamedTuple

from pyrado.algorithms.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.algorithms.utils import ReplayMemory
from pyrado.policies.dummy import DummyPolicy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.data_format import to_format
from pyrado.sampling.step_sequence import discounted_value, gae_returns
from pyrado.sampling.rollout import rollout
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim


rewards = [
    -200,
    -100,
    -50,
    -25,
    -17.5,
]
# Observations has one additional element
observations = [
    np.array([3, 2, 7]),
    np.array([3, 1, 7]),
    np.array([2, 0, 7]),
    np.array([3, 1, 3]),
    np.array([0, 2, 4]),
    np.array([1, 1, 1]),
]
# Actions come from PyTorch
actions = [
    to.tensor([0, 1]),
    to.tensor([0, 3]),
    to.tensor([2, 4]),
    to.tensor([3, 1]),
    to.tensor([0, 0]),
]
# Policy infos as dict collapse test
policy_infos = [
    {'mean': np.array([0, 1]), 'std': 0.4},
    {'mean': np.array([0, 3]), 'std': 0.2},
    {'mean': np.array([2, 4]), 'std': 0.1},
    {'mean': np.array([3, 1]), 'std': 0.05},
    {'mean': np.array([0, 0]), 'std': 0.025},
]
# Hidden is a tuple, like we see with LSTMs
hidden = [
    (np.array([3, 2, 7]), np.array([2, 1])),
    (np.array([4, 9, 8]), np.array([5, 6])),
    (np.array([1, 4, 9]), np.array([7, 3])),
    (np.array([0, 8, 2]), np.array([4, 9])),
    (np.array([2, 7, 6]), np.array([8, 0])),
]


def test_create_rew_only():
    # Don't require additional fields for this test
    StepSequence.required_fields = {}

    ro = StepSequence(rewards=rewards, data_format='numpy')
    assert len(ro) == 5
    assert (ro.rewards == np.array(rewards)).all()


@pytest.mark.parametrize(
    'data_format, tensor_type', [('numpy', np.ndarray), ('torch', to.Tensor)], ids=['numpy', 'torch']
)
def test_create(data_format, tensor_type):
    # With actions, observations and dicts
    ro = StepSequence(rewards=rewards, observations=observations, actions=actions, policy_infos=policy_infos,
                      hidden=hidden, data_format=data_format)
    assert len(ro) == 5

    assert isinstance(ro.rewards, tensor_type)
    assert isinstance(ro.observations, tensor_type)
    assert isinstance(ro.actions, tensor_type)
    assert isinstance(ro.policy_infos['mean'], tensor_type)
    assert isinstance(ro.policy_infos['std'], tensor_type)
    assert isinstance(ro.hidden[0], tensor_type)

    # Done should always be a ndarray
    assert isinstance(ro.done, np.ndarray)
    assert not ro.done[:-1].any()
    assert ro.done[-1]


@pytest.mark.parametrize(
    'other_format, tensor_type', [('torch', np.ndarray), ('numpy', to.Tensor)],
    ids=['numpy to torch', 'torch to numpy']
)
def test_convert(other_format, tensor_type):
    ro = StepSequence(rewards=rewards, observations=observations, actions=actions, policy_infos=policy_infos,
                      hidden=hidden, data_format=other_format)
    # convert
    if other_format == 'numpy':
        ro.torch()
    elif other_format == 'torch':
        ro.numpy()
    # Verify
    assert isinstance(ro.rewards, tensor_type)
    assert isinstance(ro.observations, tensor_type)
    assert isinstance(ro.actions, tensor_type)
    assert isinstance(ro.policy_infos['mean'], tensor_type)
    assert isinstance(ro.policy_infos['std'], tensor_type)
    assert isinstance(ro.hidden[0], tensor_type)

    # Done should always be a ndarray
    assert isinstance(ro.done, np.ndarray)


@pytest.mark.parametrize(
    'data_format', ['numpy', 'torch']
)
def test_step_iter(data_format):
    ro = StepSequence(rewards=rewards, observations=observations, actions=actions, policy_infos=policy_infos,
                      hidden=hidden, data_format=data_format)

    assert len(ro) == 5

    for i, step in enumerate(ro):
        assert step.reward == rewards[i]
        # Check current and next
        assert (step.observation == to_format(observations[i], data_format)).all()
        assert (step.next_observation == to_format(observations[i + 1], data_format)).all()
        # Check dict sub element
        assert (step.policy_info.mean == to_format(policy_infos[i]['mean'], data_format)).all()
        assert (step.hidden[0] == to_format(hidden[i][0], data_format)).all()


@pytest.mark.parametrize(
    'sls', [slice(2, 4), slice(2, 5, 2), slice(3), slice(4, None)]
)
@pytest.mark.parametrize(
    'data_format', ['numpy', 'torch']
)
def test_slice(sls, data_format):
    ro = StepSequence(rewards=rewards, observations=observations, actions=actions, policy_infos=policy_infos,
                      hidden=hidden, data_format=data_format)

    # Slice rollout
    sliced = ro[sls]
    # Slice reward list for verification
    sliced_rew = rewards[sls]

    for i, step in enumerate(sliced):
        assert step.reward == sliced_rew[i]


@pytest.mark.parametrize(
    'data_format', ['numpy', 'torch']
)
def test_add_data(data_format):
    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        actions=actions,
        policy_infos=policy_infos,
        hidden=hidden,
        data_format=data_format
    )
    # Add a data field
    ro.add_data('return', discounted_value(ro, 0.9))
    assert hasattr(ro, 'return')

    # Query new data field from steps
    assert abs(ro[2]['return'] - -86.675) < 0.01


@pytest.mark.parametrize(
    'data_format', ['numpy', 'torch']
)
def test_concat(data_format):
    # Create some rollouts with random rewards
    ros = [
        StepSequence(
            rewards=np.random.randn(5),
            observations=np.random.randn(6),
            actions=np.random.randn(5),
            policy_infos={'mean': np.random.randn(5)},
            hidden=(np.random.randn(5), np.random.randn(5)),
            data_format=data_format
        ),
        StepSequence(
            rewards=np.random.randn(5),
            observations=np.random.randn(6),
            actions=np.random.randn(5),
            policy_infos={'mean': np.random.randn(5)},
            hidden=(np.random.randn(5), np.random.randn(5)),
            data_format=data_format
        )
    ]

    # Perform concatenation
    cat = StepSequence.concat(ros)

    assert cat.continuous
    assert cat.rollout_count == 2

    # Check steps
    for step_ro, step_cat in zip(itertools.chain.from_iterable(ros), cat):
        assert step_ro.reward == step_cat.reward
        assert step_ro.observation == step_cat.observation
        assert step_ro.done == step_cat.done


@pytest.mark.parametrize(
    'data_format', ['numpy', 'torch']
)
def test_split_multi(data_format):
    # Don't require additional fields for this test
    StepSequence.required_fields = {}

    ro = StepSequence(
        rewards=np.arange(20),
        rollout_bounds=[0, 4, 11, 17, 20],
        data_format=data_format
    )

    # There should be four parts
    assert ro.rollout_count == 4
    # Of these sizes
    assert list(ro.rollout_lengths) == [4, 7, 6, 3]

    # Test selecting one
    s1 = ro.get_rollout(1)
    assert s1.rollout_count == 1
    assert s1[0].reward == ro[4].reward

    # Test selecting a slice
    s2 = ro.get_rollout(slice(1, -1))
    assert s2.rollout_count == 2
    assert s2[0].reward == ro[4].reward
    assert s2[7].reward == ro[11].reward

    # Test selecting by list
    s2 = ro.get_rollout([1, 3])
    assert s2.rollout_count == 2
    assert s2[0].reward == ro[4].reward
    assert s2[7].reward == ro[17].reward


@pytest.mark.parametrize(
    'data_format', ['numpy', 'torch']
)
def test_pickle(data_format):
    ro = StepSequence(rewards=rewards, observations=observations, actions=actions, policy_infos=policy_infos,
                      hidden=hidden, data_format=data_format)

    # Pickle/unpickle
    ro2 = pickle.loads(pickle.dumps(ro, pickle.HIGHEST_PROTOCOL))

    for step, step_pi in zip(ro, ro2):
        assert step.reward == step_pi.reward
        assert (step.observation == step_pi.observation).all()
        assert (step.action == step_pi.action).all()
        assert step.done == step_pi.done


@pytest.mark.parametrize(
    'env', [
        BallOnBeamSim(dt=0.01, max_steps=200),
    ], ids=['bob_linpol']
)
def test_advantage_calculation(env, linear_policy):
    ro = rollout(env, linear_policy)
    gamma = 0.99
    lamb = 0.95

    # Add dummy values
    values = np.ones_like(ro.rewards)
    if not ro.done[-1]:
        values = to.cat([values, 0])
    ro.add_data('values', values)

    gae1 = gae_returns(ro, gamma, lamb)

    # Compute the advantages
    gae2 = np.empty_like(values)
    for k in reversed(range(ro.length)):
        if ro[k].done:
            gae2[k] = ro[k].reward - values[k]
        else:
            gae2[k] = ro[k].reward + gamma*values[k + 1] - values[k] + \
                      gamma*lamb*gae2[k + 1]

    assert (gae1 == gae2).all()


@pytest.mark.parametrize(
    'capacity', [
        1, 2, 8,
    ], ids=['1', '2', '8']
)
def test_replay_memory(capacity):
    rm = ReplayMemory(capacity)

    # Create fake rollouts (of length 5)
    ro1 = StepSequence(rewards=rewards, observations=observations, actions=actions, hidden=hidden)
    ro2 = StepSequence(rewards=rewards, observations=observations, actions=actions, hidden=hidden)
    # Concatenate them for testing only
    ros = StepSequence.concat([ro1, ro2], truncate_last=True)  # same truncate_last behavior as push function

    # Check the lengths
    rm.push(ro1)
    assert len(rm) == len(ro1) or len(rm) == capacity
    rm.push(ro2)
    assert len(rm) == len(ro1) + len(ro1) or len(rm) == capacity

    # Check the elements
    shift = len(ros) - capacity
    if shift < len(ro1):
        assert all(rm.memory.observations[0] == ros.observations[shift])
    assert all(rm.memory.observations[-1] == ro2.observations[-2])  # -2 since one was truncated


# A dummy namedtuple for testing
class DummyNT(NamedTuple):
    part1: to.Tensor
    part2: to.Tensor


@pytest.mark.parametrize(
    'data_format', ['numpy', 'torch']
)
def test_namedtuple(data_format):
    hid_nt = [DummyNT(*it) for it in hidden]

    ro = StepSequence(
        rewards=rewards,
        hidden=hid_nt,
        data_format=data_format
    )

    assert isinstance(ro.hidden, DummyNT)

    for i, step in enumerate(ro):
        assert isinstance(step.hidden, DummyNT)
        assert (step.hidden.part1 == to_format(hid_nt[i].part1, data_format)).all()


@pytest.mark.parametrize(
    'env', [
        'default_pend',
        'default_bob',
    ], ids=['pend', 'bob'],
    indirect=True
)
@pytest.mark.parametrize(
    'num_real_ros', [1, 3], ids=['1realro', '3realro']
)
@pytest.mark.parametrize(
    'num_sim_ros', [1, 3], ids=['1simro', '3simro']
)
@pytest.mark.parametrize(
    'max_real_steps, max_sim_steps',
    [
        (4, 4,), (4, 7), (7, 4), (10000, 10000)
    ], ids=['real=sim', 'real<sim', 'real>sim', 'inf']
)
def test_truncate_rollouts(env, num_real_ros, num_sim_ros, max_real_steps, max_sim_steps):
    policy = DummyPolicy(env.spec)
    ros_real = []
    ros_sim = []

    # Create the rollout data
    for _ in range(num_real_ros):
        ros_real.append(rollout(env, policy, eval=True, max_steps=max_real_steps, stop_on_done=True))
    for _ in range(num_sim_ros):
        ros_sim.append(rollout(env, policy, eval=True, max_steps=max_sim_steps, stop_on_done=True))

    # Truncate them
    ros_real_tr, ros_sim_tr = SysIdViaEpisodicRL.truncate_rollouts(ros_real, ros_sim)

    # Obtained the right number of rollouts
    assert len(ros_real_tr) == len(ros_sim_tr)

    for ro_r, ro_s in zip(ros_real_tr, ros_sim_tr):
        # All individual truncated rollouts have the correct length
        assert ro_r.length == ro_s.length
