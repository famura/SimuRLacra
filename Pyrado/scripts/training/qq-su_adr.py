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
Train an agent to solve the Quanser Qube swing-up task using Soft Actor Critic.
"""
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.meta.adr import ADR
from pyrado.algorithms.step_based.gae import GAE, ValueFunctionSpace
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger import step
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.feed_back.two_headed_fnn import TwoHeadedFNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.spaces.box import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    seed_str = f"seed-{args.seed}" if args.seed is not None else None

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{ADR.name}", seed_str)

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 100.0, max_steps=600)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    ### Subroutine ###

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Critic
    vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
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
    subroutine_hparam = dict(
        max_iter=200,
        eps_clip=0.12648736789309026,
        min_steps=30 * env.max_steps,
        num_epoch=10,
        batch_size=500,
        std_init=0.7573286998997557,
        lr=6.999956625305722e-04,
        max_grad_norm=1.0,
        num_workers=8,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    subroutine = PPO(ex_dir, env, policy, critic, **subroutine_hparam)

    ### ADR ###

    adr_hp = dict(randomized_params=["length_rot_pole", "gravity_const"], evaluation_steps=20, step_length=0.05)

    svpg_hp = {
        "actor": dict(
            hidden_sizes=[64],
            hidden_nonlin=to.relu,
        ),
        "vfcn": dict(
            hidden_sizes=[32],
            hidden_nonlin=to.tanh,
        ),
        "critic": dict(
            gamma=0.99,
            lamda=0.95,
            num_epoch=5,
            lr=1e-3,
            standardize_adv=False,
            max_grad_norm=5.0,
        ),
        "particle": dict(max_iter=200, min_steps=2 * env.max_steps, lr=5e-4, num_workers=1),
        "algo": dict(max_iter=200, num_particles=15, temperature=10, horizon=50),
    }

    reward_generator_hp = dict(batch_size=64, reward_multiplier=1.0)

    rest_hp = dict(max_iter=30, num_discriminator_epoch=10, batch_size=64, num_trajs_per_config=50)

    adr = ADR(ex_dir, env, subroutine, adr_hp, svpg_hp, reward_generator_hp, **rest_hp)

    # Policy
    policy_hparam = dict(shared_hidden_sizes=[64, 64], shared_hidden_nonlin=to.relu)
    policy = TwoHeadedFNNPolicy(spec=env.spec, **policy_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        save_dir=ex_dir,
    )

    # Jeeeha
    adr.train(snapshot_mode="latest", seed=args.seed)
