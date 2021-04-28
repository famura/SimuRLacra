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
Train an agent to solve the Qube swing-up task using Bayesian Domain Randomization.
"""
import numpy as np
import torch as to
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.domain_randomization.default_randomizers import (
    create_default_domain_param_map_qq,
    create_zero_var_randomizer,
)
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.quanser.quanser_qube import QQubeSwingUpReal
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.spaces import BoxSpace, ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QQubeSwingUpSim.name, f"{BayRn.name}-{PPO.name}_{FNNPolicy.name}", "rand-Mp-Mr-Lp-Lr_lower-std"
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 100.0, max_steps=600)
    env_sim = QQubeSwingUpSim(**env_sim_hparams)
    env_sim = ActNormWrapper(env_sim)
    env_sim = DomainRandWrapperLive(env_sim, create_zero_var_randomizer(env_sim))
    dp_map = create_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    env_real_hparams = dict(dt=1 / 500.0, max_steps=3000)
    env_real = QQubeSwingUpReal(**env_real_hparams)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env_sim.spec, **policy_hparam)

    # Critic
    vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    vfcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **vfcn_hparam)
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
    subrtn_hparam = dict(
        max_iter=200,
        eps_clip=0.12648736789309026,
        min_steps=30 * env_sim.max_steps,
        num_epoch=7,
        batch_size=500,
        std_init=0.7573286998997557,
        lr=6.999956625305722e-04,
        max_grad_norm=1.0,
        num_workers=8,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    subrtn = PPO(ex_dir, env_sim, policy, critic, **subrtn_hparam)

    # Set the boundaries for the GP
    dp_nom = QQubeSwingUpSim.get_nominal_domain_param()
    ddp_space = BoxSpace(
        bound_lo=np.array(
            [
                0.8 * dp_nom["Mp"],
                dp_nom["Mp"] / 5000,
                0.8 * dp_nom["Mr"],
                dp_nom["Mr"] / 5000,
                0.8 * dp_nom["Lp"],
                dp_nom["Lp"] / 5000,
                0.8 * dp_nom["Lr"],
                dp_nom["Lr"] / 5000,
            ]
        ),
        bound_up=np.array(
            [
                1.2 * dp_nom["Mp"],
                dp_nom["Mp"] / 20,
                1.2 * dp_nom["Mr"],
                dp_nom["Mr"] / 20,
                1.2 * dp_nom["Lp"],
                dp_nom["Lp"] / 20,
                1.2 * dp_nom["Lr"],
                dp_nom["Lr"] / 20,
            ]
        ),
    )

    # policy_init = to.load(osp.join(pyrado.EXP_DIR, QQubeSwingUpSim.name, PPO.name, 'EXP_NAME', 'policy.pt'))
    # valuefcn_init = to.load(osp.join(pyrado.EXP_DIR, QQubeSwingUpSim.name, PPO.name, 'EXP_NAME', 'valuefcn.pt'))

    # Algorithm
    bayrn_hparam = dict(
        thold_succ=300.0,
        max_iter=15,
        acq_fc="EI",
        # acq_param=dict(beta=0.2),
        acq_restarts=500,
        acq_samples=1000,
        num_init_cand=10,
        warmstart=False,
        # policy_param_init=policy_init.param_values.data,
        # valuefcn_param_init=valuefcn_init.param_values.data,
    )

    # Save the environments and the hyper-parameters
    save_dicts_to_yaml(
        dict(env_sim=env_sim_hparams, env_real=env_real_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, vfcn=vfcn_hparam),
        dict(subrtn=subrtn_hparam, subrtn_name=PPO.name),
        dict(algo=bayrn_hparam, algo_name=BayRn.name, dp_map=dp_map),
        save_dir=ex_dir,
    )

    algo = BayRn(ex_dir, env_sim, env_real, subrtn, ddp_space=ddp_space, **bayrn_hparam)

    # Jeeeha
    algo.train(
        snapshot_mode="latest",
        seed=args.seed,
    )

    # Train the policy on the most lucrative domain
    BayRn.train_argmax_policy(
        ex_dir,
        env_sim,
        ppo,
        num_restarts=500,
        num_samples=1000,
        # policy_param_init=policy.param_values.data,
        # valuefcn_param_init=critic.vfcn.param_values.data
    )
