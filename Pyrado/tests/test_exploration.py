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

from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.exploration.stochastic_params import NormalParamNoise
from pyrado.policies.features import *
from pyrado.exploration.stochastic_action import NormalActNoiseExplStrat


@pytest.mark.parametrize(
    'env', [
        BallOnBeamSim(dt=0.02, max_steps=1),
        QBallBalancerSim(dt=0.02, max_steps=1),
    ],
    ids=['bob', 'qbb']
)
@pytest.mark.parametrize(
    'policy',
    ['linear_policy', 'fnn_policy'],
    ids=['lin', 'fnn'],
    indirect=True
)
def test_noise_on_act(env, policy):
    for _ in range(100):
        # Init the exploration strategy
        act_noise_strat = NormalActNoiseExplStrat(
            policy,
            std_init=0.5,
            train_mean=True
        )

        # Set new parameters for the exploration noise
        std = to.ones(env.act_space.flat_dim)*to.rand(1)
        mean = to.rand(env.act_space.shape)
        act_noise_strat.noise.adapt(mean, std)
        assert (mean == act_noise_strat.noise.mean).all()

        # Sample a random observation from the environment
        obs = to.from_numpy(env.obs_space.sample_uniform())

        # Get a clean and a noisy action
        act = policy(obs)  # policy expects Tensors
        act_noisy = act_noise_strat(obs)  # exploration strategy expects ndarrays
        assert isinstance(act, to.Tensor)
        assert not to.equal(act, act_noisy)


@pytest.mark.parametrize(
    'env', [
        BallOnBeamSim(dt=0.02, max_steps=1),
        QBallBalancerSim(dt=0.02, max_steps=1),
    ],
    ids=['bob', 'qbb']
)
@pytest.mark.parametrize(
    'policy',
    ['linear_policy', 'fnn_policy'],
    ids=['lin', 'fnn'],
    indirect=True
)
def test_noise_on_param(env, policy):
    for _ in range(5):
        # Init the exploration strategy
        param_noise_strat = NormalParamNoise(
            policy.num_param,
            full_cov=True,
            std_init=1.,
            std_min=0.01,
            train_mean=True
        )

        # Set new parameters for the exploration noise
        mean = to.rand(policy.num_param)
        cov = to.eye(policy.num_param)
        param_noise_strat.adapt(mean, cov)
        to.testing.assert_allclose(mean, param_noise_strat.noise.mean)

        # Reset exploration strategy
        param_noise_strat.reset_expl_params()

        # Sample a random observation from the environment
        obs = to.from_numpy(env.obs_space.sample_uniform())

        # Get a clean and a noisy action
        act = policy(obs)  # policy expects Tensors
        sampled_param = param_noise_strat.sample_param_set(policy.param_values)
        new_policy = deepcopy(policy)
        new_policy.param_values = sampled_param
        act_noisy = new_policy(obs)  # exploration strategy expects ndarrays

        assert isinstance(act, to.Tensor)
        assert not to.equal(act, act_noisy)
