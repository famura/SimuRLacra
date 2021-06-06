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
import itertools
import pickle
from typing import NamedTuple, Union

import numpy as np
import pandas as pd
import pytest
import torch as to
from scipy import signal
from tests.environment_wrappers.mock_env import MockEnv

from pyrado.algorithms.episodic.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.algorithms.utils import ReplayMemory
from pyrado.policies.feed_forward.dummy import DummyPolicy
from pyrado.sampling.data_format import to_format
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence, discounted_value, gae_returns
from pyrado.spaces.box import InfBoxSpace


@pytest.fixture
def mock_data():
    rewards = [
        -200.0,
        -100,
        -50,
        -25,
        -17.5,
    ]
    # Observations has one additional element
    observations = [
        np.array([3.0, 2, 7, 5], dtype=np.float64),
        np.array([3.0, 1, 9, 5], dtype=np.float64),
        np.array([2.0, 0, 7, 5], dtype=np.float64),
        np.array([3.0, 1, 3, 5], dtype=np.float64),
        np.array([0.0, 2, 4, 5], dtype=np.float64),
        np.array([1.0, 8, 1, 5], dtype=np.float64),
    ]
    # States has one additional element
    states = [
        np.array([4.0, 8, 7], dtype=np.float64),
        np.array([2.0, 1, 7], dtype=np.float64),
        np.array([1.0, 0, 7], dtype=np.float64),
        np.array([4.0, 1, 7], dtype=np.float64),
        np.array([0.0, 2, 7], dtype=np.float64),
        np.array([0.0, 1, 7], dtype=np.float64),
    ]
    # Actions come from PyTorch
    actions = [
        to.tensor([0.0, 1], dtype=to.get_default_dtype()),
        to.tensor([0.0, 3], dtype=to.get_default_dtype()),
        to.tensor([2.0, 4], dtype=to.get_default_dtype()),
        to.tensor([3.0, 1], dtype=to.get_default_dtype()),
        to.tensor([0.0, 0], dtype=to.get_default_dtype()),
    ]
    # Policy infos as dict collapse test
    policy_infos = [
        {"mean": np.array([0.0, 1], dtype=np.float64), "std": 0.4},
        {"mean": np.array([0.0, 3], dtype=np.float64), "std": 0.2},
        {"mean": np.array([2.0, 4], dtype=np.float64), "std": 0.1},
        {"mean": np.array([3.0, 1], dtype=np.float64), "std": 0.05},
        {"mean": np.array([0.0, 0], dtype=np.float64), "std": 0.025},
    ]
    # Hidden is a tuple, like we see with LSTMs
    hidden = [
        (to.tensor([3.0, 2, 7], dtype=to.get_default_dtype()), to.tensor([2.0, 1, 8], dtype=to.get_default_dtype())),
        (to.tensor([4.0, 9, 8], dtype=to.get_default_dtype()), to.tensor([5.0, 6, 5], dtype=to.get_default_dtype())),
        (to.tensor([1.0, 4, 9], dtype=to.get_default_dtype()), to.tensor([7.0, 3, 5], dtype=to.get_default_dtype())),
        (to.tensor([0.0, 8, 2], dtype=to.get_default_dtype()), to.tensor([4.0, 9, 3], dtype=to.get_default_dtype())),
        (to.tensor([2.0, 7, 6], dtype=to.get_default_dtype()), to.tensor([8.0, 0, 1], dtype=to.get_default_dtype())),
    ]
    return rewards, states, observations, actions, hidden, policy_infos


def test_additional_required(mock_data):
    # Require the states as additional field for this test
    StepSequence.required_fields = {"states"}

    rewards, states, observations, actions, hidden, policy_infos = mock_data

    with pytest.raises(Exception) as err:
        # This should fail
        _ = StepSequence(rewards=rewards, observations=observations, actions=actions)
        assert isinstance(err, ValueError)

    ro = StepSequence(rewards=rewards, observations=observations, actions=actions, states=states)
    assert len(ro) == 5
    assert (ro.rewards == np.array(rewards)).all()


@pytest.mark.parametrize(
    "data_format, tensor_type", [("numpy", np.ndarray), ("torch", to.Tensor)], ids=["numpy", "torch"]
)
def test_create(mock_data, data_format, tensor_type):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    # With actions, observations and dicts
    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        states=states,
        actions=actions,
        policy_infos=policy_infos,
        hidden=hidden,
        data_format=data_format,
    )
    assert len(ro) == 5

    assert isinstance(ro.rewards, tensor_type)
    assert isinstance(ro.observations, tensor_type)
    assert isinstance(ro.actions, tensor_type)
    assert isinstance(ro.policy_infos["mean"], tensor_type)
    assert isinstance(ro.policy_infos["std"], tensor_type)
    assert isinstance(ro.hidden[0], tensor_type)

    # Done should always be a ndarray
    assert isinstance(ro.done, np.ndarray)
    assert not ro.done[:-1].any()
    assert ro.done[-1]


@pytest.mark.parametrize(
    "other_format, tensor_type", [("torch", np.ndarray), ("numpy", to.Tensor)], ids=["numpy to torch", "torch to numpy"]
)
def test_convert(mock_data, other_format, tensor_type):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        states=states,
        actions=actions,
        policy_infos=policy_infos,
        hidden=hidden,
        data_format=other_format,
    )
    # convert
    if other_format == "numpy":
        ro.torch()
    elif other_format == "torch":
        ro.numpy()
    # Verify
    assert isinstance(ro.rewards, tensor_type)
    assert isinstance(ro.observations, tensor_type)
    assert isinstance(ro.actions, tensor_type)
    assert isinstance(ro.policy_infos["mean"], tensor_type)
    assert isinstance(ro.policy_infos["std"], tensor_type)
    assert isinstance(ro.hidden[0], tensor_type)

    # Done should always be a ndarray
    assert isinstance(ro.done, np.ndarray)


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_step_iter(mock_data, data_format: str):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        states=states,
        actions=actions,
        policy_infos=policy_infos,
        hidden=hidden,
        data_format=data_format,
    )

    assert len(ro) == 5

    for i, step in enumerate(ro):
        assert step.reward == rewards[i]
        # Check current and next
        assert (step.observation == to_format(observations[i], data_format)).all()
        assert (step.next_observation == to_format(observations[i + 1], data_format)).all()
        # Check dict sub element
        assert (step.policy_info.mean == to_format(policy_infos[i]["mean"], data_format)).all()
        assert (step.hidden[0] == to_format(hidden[i][0], data_format)).all()


@pytest.mark.parametrize("sls", [slice(2, 4), slice(2, 5, 2), slice(3), slice(4, None)])
@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_slice(mock_data, sls, data_format: str):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        states=states,
        actions=actions,
        policy_infos=policy_infos,
        hidden=hidden,
        data_format=data_format,
    )

    # Slice rollout
    sliced = ro[sls]
    # Slice reward list for verification
    sliced_rew = rewards[sls]

    for i, step in enumerate(sliced):
        assert step.reward == sliced_rew[i]


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_add_data(mock_data, data_format: str):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        states=states,
        actions=actions,
        policy_infos=policy_infos,
        hidden=hidden,
        data_format=data_format,
    )
    # Add a data field
    ro.add_data("return", discounted_value(ro, 0.9))
    assert hasattr(ro, "return")

    # Query new data field from steps
    assert abs(ro[2]["return"] - -86.675) < 0.01


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_concat(data_format: str):
    # Create some rollouts with random rewards
    ros = [
        StepSequence(
            rewards=np.random.randn(5),
            observations=np.random.randn(6),
            states=np.random.randn(6),
            actions=np.random.randn(5),
            policy_infos={"mean": np.random.randn(5)},
            hidden=(np.random.randn(5), np.random.randn(5)),
            data_format=data_format,
        ),
        StepSequence(
            rewards=np.random.randn(5),
            observations=np.random.randn(6),
            states=np.random.randn(6),
            actions=np.random.randn(5),
            policy_infos={"mean": np.random.randn(5)},
            hidden=(np.random.randn(5), np.random.randn(5)),
            data_format=data_format,
        ),
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


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_split_multi(data_format: str):
    # Don't require additional fields for this test
    StepSequence.required_fields = {}

    ro = StepSequence(
        rewards=np.arange(20),
        rollout_bounds=[0, 4, 11, 17, 20],
        observations=np.empty(21),
        actions=np.empty(20),
        data_format=data_format,
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


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_pickle(mock_data, data_format: str):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        actions=actions,
        policy_infos=policy_infos,
        hidden=hidden,
        data_format=data_format,
    )

    # Pickle/unpickle
    ro2 = pickle.loads(pickle.dumps(ro, pickle.HIGHEST_PROTOCOL))

    for step, step_pi in zip(ro, ro2):
        assert step.reward == step_pi.reward
        assert (step.observation == step_pi.observation).all()
        assert (step.action == step_pi.action).all()
        assert step.done == step_pi.done


@pytest.mark.parametrize(
    ["env", "policy"],
    [
        ("default_bob", "linear_policy"),
    ],
    ids=["bob_linpol"],
    indirect=True,
)
def test_advantage_calculation(env, policy):
    ro = rollout(env, policy)
    gamma = 0.99
    lamb = 0.95

    # Add dummy values
    values = np.ones_like(ro.rewards)
    if not ro.done[-1]:
        values = to.cat([values, 0])
    ro.add_data("values", values)

    gae1 = gae_returns(ro, gamma, lamb)

    # Compute the advantages
    gae2 = np.empty_like(values)
    for k in reversed(range(ro.length)):
        if ro[k].done:
            gae2[k] = ro[k].reward - values[k]
        else:
            gae2[k] = ro[k].reward + gamma * values[k + 1] - values[k] + gamma * lamb * gae2[k + 1]

    assert (gae1 == gae2).all()


@pytest.mark.parametrize(
    "capacity",
    [
        1,
        2,
        8,
    ],
    ids=["1", "2", "8"],
)
def test_replay_memory(mock_data, capacity):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    rm = ReplayMemory(capacity)

    # Create fake rollouts (of length 5)
    ro1 = StepSequence(rewards=rewards, observations=observations, states=states, actions=actions, hidden=hidden)
    ro2 = StepSequence(rewards=rewards, observations=observations, states=states, actions=actions, hidden=hidden)
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


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_namedtuple(mock_data, data_format: str):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    hid_nt = [DummyNT(*it) for it in hidden]

    ro = StepSequence(
        rewards=rewards, actions=actions, observations=observations, hidden=hid_nt, data_format=data_format
    )

    assert isinstance(ro.hidden, DummyNT)

    for i, step in enumerate(ro):
        assert isinstance(step.hidden, DummyNT)
        assert (step.hidden.part1 == to_format(hid_nt[i].part1, data_format)).all()


@pytest.mark.parametrize(
    "env",
    [
        "default_pend",
        "default_bob",
    ],
    ids=["pend", "bob"],
    indirect=True,
)
@pytest.mark.parametrize("num_real_ros", [1, 3], ids=["1realro", "3realro"])
@pytest.mark.parametrize("num_sim_ros", [1, 3], ids=["1simro", "3simro"])
@pytest.mark.parametrize(
    "max_real_steps, max_sim_steps",
    [
        (
            4,
            4,
        ),
        (4, 7),
        (7, 4),
        (10000, 10000),
    ],
    ids=["real=sim", "real<sim", "real>sim", "inf"],
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


@pytest.mark.parametrize("data_format", ["numpy", "torch"])
def test_process(mock_data, data_format: str):
    rewards, states, observations, actions, hidden, policy_infos = mock_data

    # Create the rollout
    ro = StepSequence(rewards=rewards, observations=observations, states=states, actions=actions, hidden=hidden)

    if data_format == "numpy":
        # Create the filter (arbitrary values)
        b, a = signal.butter(N=5, Wn=10, fs=100)

        # Filter the signals, but not the time
        ro_proc = StepSequence.process_data(
            ro, signal.filtfilt, fcn_arg_name="x", exclude_fields=["time"], b=b, a=a, padlen=2, axis=0
        )

    else:
        # Transform to PyTorch data and define a simple function
        ro.torch()
        ro_proc = StepSequence.process_data(
            ro, lambda x: x * 2, fcn_arg_name="x", include_fields=["time"], fcn_arg_types=to.Tensor
        )

    assert isinstance(ro_proc, StepSequence)
    assert ro_proc.length == ro.length


@pytest.mark.parametrize("given_rewards", [True, False], ids=["rewards", "norewards"])
def test_stepsequence_from_pandas(mock_data, given_rewards: bool):
    rewards, states, observations, actions, hidden, policy_infos = mock_data
    states = np.asarray(states)
    observations = np.asarray(observations)
    actions = to.stack(actions).numpy()
    rewards = np.asarray(rewards)

    # Create fake observed data set. The labels must match the labels of the spaces. The order can be mixed.
    content = dict(
        s0=states[:, 0],
        s1=states[:, 1],
        s2=states[:, 2],
        o3=observations[:, 3],
        o0=observations[:, 0],
        o2=observations[:, 2],
        o1=observations[:, 1],
        a1=actions[:, 1],
        a0=actions[:, 0],
        # Some content that was not in
        steps=np.arange(0, states.shape[0]),
        infos=[dict(foo="bar")] * 6,
    )
    if given_rewards:
        content["rewards"] = rewards
    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in content.items()]))

    env = MockEnv(
        state_space=InfBoxSpace(shape=states[0].shape, labels=["s0", "s1", "s2"]),
        obs_space=InfBoxSpace(shape=observations[0].shape, labels=["o0", "o1", "o2", "o3"]),
        act_space=InfBoxSpace(shape=actions[0].shape, labels=["a0", "a1"]),
    )

    reconstructed = StepSequence.from_pandas(df, env.spec)

    assert len(reconstructed.rewards) == len(rewards)
    assert np.allclose(reconstructed.states, states)
    assert np.allclose(reconstructed.observations, observations)
    assert np.allclose(reconstructed.actions, actions)


@pytest.mark.parametrize("data_format", ["numpy", "torch"], ids=["numpy", "torch"])
@pytest.mark.parametrize("pad_value", [0, 0.14], ids=["zero", "somefloat"])
def test_stepsequence_padding(mock_data, data_format: str, pad_value: Union[int, float]):
    # Create too short rollout
    rewards, states, observations, actions, hidden, policy_infos = mock_data
    ro = StepSequence(
        rewards=rewards,
        observations=observations,
        states=states,
        actions=actions,
        hidden=hidden,
        policy_infos=policy_infos,
    )
    len_orig = ro.length

    if data_format == "torch":
        ro.torch()

    # Pad it
    StepSequence.pad(ro, len_to_pad_to=len(ro) + 7, pad_value=pad_value)

    # Check
    ro.numpy()  # for simplified checking
    assert np.allclose(ro.states[len_orig + 1 :], pad_value * np.ones_like(ro.states[len_orig + 1 :]))
    assert np.allclose(ro.observations[len_orig + 1 :], pad_value * np.ones_like(ro.observations[len_orig + 1 :]))
    assert np.allclose(ro.actions[len_orig:], pad_value * np.ones_like(ro.actions[len_orig:]))
    assert np.allclose(ro.rewards[len_orig:], pad_value * np.ones_like(ro.rewards[len_orig:]))
    for k, v in ro.policy_infos.items():
        assert np.allclose(v[len_orig:], pad_value * np.ones_like(v[len_orig:]))
    # assert ro.length == le.n_orig + 7
