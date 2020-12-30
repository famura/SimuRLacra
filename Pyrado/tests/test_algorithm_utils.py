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
import numpy as np
import torch as to
from copy import deepcopy
from torch.distributions.normal import Normal

from pyrado.algorithms.meta.adr import RewardGenerator
from pyrado.algorithms.utils import compute_action_statistics, until_thold_exceeded, get_grad_via_torch
from pyrado.domain_randomization.default_randomizers import create_default_randomizer_omo
from pyrado.environments.sim_base import SimEnv
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.policies.base import TwoHeadedPolicy, Policy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.step_sequence import StepSequence
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler


@to.no_grad()
@pytest.mark.parametrize(
    "env",
    [
        "default_pend",
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "policy",
    [
        "linear_policy",
        "fnn_policy",
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "thfnn_policy",
        "thgru_policy",
    ],
    ids=["lin", "fnn", "rnn", "lstm", "gru", "adn", "thfnn", "thgru"],
    indirect=True,
)
def test_action_statistics(env: SimEnv, policy: Policy):
    sigma = 1.0  # with lower values like 0.1 we can observe violations of the tolerances

    # Create an action-based exploration strategy
    explstrat = NormalActNoiseExplStrat(policy, std_init=sigma)

    # Sample a deterministic rollout
    ro_policy = rollout(env, policy, eval=True, max_steps=1000, stop_on_done=False, seed=0)
    ro_policy.torch()

    # Run the exploration strategy on the previously sampled rollout
    if policy.is_recurrent:
        if isinstance(policy, TwoHeadedPolicy):
            act_expl, _, _ = explstrat(ro_policy.observations)
        else:
            act_expl, _ = explstrat(ro_policy.observations)
        # Get the hidden states from the deterministic rollout
        hidden_states = ro_policy.hidden_states
    else:
        if isinstance(policy, TwoHeadedPolicy):
            act_expl, _ = explstrat(ro_policy.observations)
        else:
            act_expl = explstrat(ro_policy.observations)
        hidden_states = [0.0] * ro_policy.length  # just something that does not violate the format

    ro_expl = StepSequence(
        actions=act_expl[:-1],  # truncate act due to last obs
        observations=ro_policy.observations,
        rewards=ro_policy.rewards,  # don't care but necessary
        hidden_states=hidden_states,
    )

    # Compute action statistics and the ground truth
    actstats = compute_action_statistics(ro_expl, explstrat)
    gt_logprobs = Normal(loc=ro_policy.actions, scale=sigma).log_prob(ro_expl.actions)
    gt_entropy = Normal(loc=ro_policy.actions, scale=sigma).entropy()

    to.testing.assert_allclose(actstats.log_probs, gt_logprobs, rtol=1e-4, atol=1e-5)
    to.testing.assert_allclose(actstats.entropy, gt_entropy, rtol=1e-4, atol=1e-5)


@pytest.mark.longtime
@pytest.mark.parametrize(
    "env",
    [
        "default_omo",
    ],
    indirect=True,
)
def test_adr_reward_generator(env):
    reference_env = env
    random_env = deepcopy(env)
    reward_generator = RewardGenerator(
        env_spec=random_env.spec,
        batch_size=256,
        reward_multiplier=1,
        lr=5e-3,
    )
    policy = FNNPolicy(reference_env.spec, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    dr = create_default_randomizer_omo()
    dr.randomize(num_samples=1)
    random_env.domain_param = dr.get_params(fmt="dict", dtype="numpy")
    reference_sampler = ParallelRolloutSampler(reference_env, policy, num_workers=1, min_steps=1000)
    random_sampler = ParallelRolloutSampler(random_env, policy, num_workers=1, min_steps=1000)

    losses = []
    for i in range(200):
        reference_traj = StepSequence.concat(reference_sampler.sample())
        random_traj = StepSequence.concat(random_sampler.sample())
        losses.append(reward_generator.train(reference_traj, random_traj, 10))
    assert losses[len(losses) - 1] < losses[0]


@pytest.mark.parametrize("thold", [0.5], ids=["0.5"])
@pytest.mark.parametrize("max_iter", [None, 2], ids=["relentless", "twice"])
def test_until_thold_exceeded(thold, max_iter):
    @until_thold_exceeded(thold, max_iter)
    def _trn_eval_fcn():
        # Draw a random number to mimic a training and evaluation process
        return np.random.rand(1)

    for _ in range(10):
        val = _trn_eval_fcn()
        if max_iter is None:
            assert val >= thold
        else:
            assert True  # there is no easy way to insect the counter, read the printed messages


def test_get_grad_via_torch():
    def to_fcn(x: np.ndarray):
        return (x + 2) ** 2 * 3

    x = np.ones((2, 2))
    grad_np = get_grad_via_torch(x, to_fcn)
    assert isinstance(grad_np, np.ndarray)
    assert np.allclose(grad_np, 18 * np.ones((2, 2)))
