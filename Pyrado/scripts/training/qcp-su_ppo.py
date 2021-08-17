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
Train an agent to solve the Quanser CartPole swing-up task using Proximal Policy Optimization.
"""
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.recurrent.rnn import GRUPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if args.mode is None or args.mode.lower() == FNNPolicy.name:
        pol_str = FNNPolicy.name
    elif args.mode.lower() == GRUPolicy.name:
        pol_str = GRUPolicy.name
    else:
        raise pyrado.ValueErr(given=args.mode, eq_constraint=f"{FNNPolicy.name}, {GRUPolicy.name}, or None")
    seed_str = f"seed-{args.seed}" if args.seed is not None else None

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QCartPoleSwingUpSim.name, f"{PPO.name}_{pol_str}", seed_str)

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparam = dict(dt=1 / 100.0, max_steps=600)
    env = QCartPoleSwingUpSim(**env_hparam)
    # env = ObsVelFiltWrapper(env, idcs_pos=["theta", "alpha"], idcs_vel=["theta_dot", "alpha_dot"])
    env = ActNormWrapper(env)

    # Policy
    if args.mode is None or args.mode.lower() == FNNPolicy.name:
        policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
        policy = FNNPolicy(spec=env.spec, **policy_hparam)
    elif args.mode.lower() == GRUPolicy.name:
        policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)
        policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    if isinstance(policy, FNNPolicy):
        vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.relu)  # FNN
        vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    elif isinstance(policy, GRUPolicy):
        vfcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
        vfcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.9844224855479998,
        lamda=0.9700148505302241,
        num_epoch=5,
        batch_size=500,
        standardize_adv=False,
        lr=7.058326426522811e-4,
        max_grad_norm=6.0,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    critic = GAE(vfcn, **critic_hparam)

    # Subroutine
    algo_hparam = dict(
        max_iter=200 if policy.name == FNNPolicy.name else 75,
        eps_clip=0.12648736789309026,
        min_steps=30 * env.max_steps,
        num_epoch=7,
        batch_size=500,
        std_init=0.7573286998997557,
        lr=6.999956625305722e-04,
        max_grad_norm=1.0,
        num_workers=8,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparam, seed=args.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, vfcn=vfcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="latest", seed=args.seed)
