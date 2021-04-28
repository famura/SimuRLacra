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
Train an agent to solve the Ball-on-Beam environment using Asynchronous Actor-Critic.
"""
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.step_based.a2c import A2C
from pyrado.algorithms.step_based.gae import GAE
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.features import FeatureStack, RBFFeat, RFFeat, identity_feat, sin_feat
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment
    ex_dir = setup_experiment(BallOnBeamSim.name, f"{A2C.name}_{LinearPolicy.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 100.0, max_steps=500)
    env = BallOnBeamSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        # feats=FeatureStack(
        #     [
        #         RFFeat(
        #             env.obs_space.flat_dim, num_feat_per_dim=500, bandwidth=1/env.obs_space.bound_up, use_cuda=True
        #         )
        #     ]
        # )
        # feats=FeatureStack([RBFFeat(num_feat_per_dim=20, bounds=env.obs_space.bounds, scale=None, use_cuda=True)])
        feats=FeatureStack([identity_feat, sin_feat])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam, use_cuda=True)

    # Critic
    vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam, use_cuda=True)
    critic_hparam = dict(
        gamma=0.99,
        lamda=0.95,
        batch_size=100,
        standardize_adv=True,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.99),
    )
    critic = GAE(vfcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=env.max_steps * 10,
        num_workers=4,
        vfcn_coeff=0.7,
        entropy_coeff=4e-5,
        batch_size=256,
        std_init=0.8,
        lr=2e-3,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.99),
    )
    algo = A2C(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, vfcn=vfcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
