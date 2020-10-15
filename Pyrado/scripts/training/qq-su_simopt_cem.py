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
Train an agent to solve the Qube swing-up task using Simulation Optimization running Cross-Entropy Method.
"""
import torch as to

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.simopt import SimOpt
from pyrado.algorithms.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.policies.domain_distribution import DomainDistrParamPolicy
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper, DomainRandWrapperLive
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f'{SimOpt.name}-{CEM.name}_{PPO.name}_{FNNPolicy.name}')
    num_workers = 8

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1/100., max_steps=600)
    env_real = QQubeReal(**env_hparams)
    env_real = ActNormWrapper(env_real)

    env_sim = QQubeSwingUpSim(**env_hparams)
    env_sim = ActNormWrapper(env_sim)
    randomizer = DomainRandomizer(
        NormalDomainParam(name='Mr', mean=0., std=1e6, clip_lo=1e-3),
        NormalDomainParam(name='Mp', mean=0., std=1e6, clip_lo=1e-3),
        NormalDomainParam(name='Lr', mean=0., std=1e6, clip_lo=1e-3),
        NormalDomainParam(name='Lp', mean=0., std=1e6, clip_lo=1e-3),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {
        0: ('Mr', 'mean'), 1: ('Mr', 'std'),
        2: ('Mp', 'mean'), 3: ('Mp', 'std'),
        4: ('Lr', 'mean'), 5: ('Lr', 'std'),
        6: ('Lp', 'mean'), 7: ('Lp', 'std')
    }
    trafo_mask = [True]*8
    # trafo_mask = [False, True, False, True, False, True, False, True]
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Subroutine for policy improvement
    behav_policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    behav_policy = FNNPolicy(spec=env_sim.spec, **behav_policy_hparam)
    value_fcn_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **value_fcn_hparam)

    critic_hparam = dict(
        gamma=0.9885,
        lamda=0.9648,
        num_epoch=2,
        batch_size=250,
        standardize_adv=False,
        lr=5.792e-4,
        max_grad_norm=1.,
    )
    critic = GAE(value_fcn, **critic_hparam)
    subrtn_policy_hparam = dict(
        max_iter=300,
        min_steps=23*env_sim.max_steps,
        num_epoch=7,
        eps_clip=0.0744,
        batch_size=250,
        std_init=0.9074,
        lr=3.446e-04,
        max_grad_norm=1.,
        num_workers=num_workers,
    )
    subrtn_policy = PPO(ex_dir, env_sim, behav_policy, critic, **subrtn_policy_hparam)

    prior = DomainRandomizer(
        NormalDomainParam(name='Mr', mean=0.095, std=0.095/10),
        NormalDomainParam(name='Mp', mean=0.024, std=0.024/10),
        NormalDomainParam(name='Lr', mean=0.085, std=0.085/10),
        NormalDomainParam(name='Lp', mean=0.129, std=0.129/10),
    )
    ddp_policy_hparam = dict(
        mapping=dp_map, trafo_mask=trafo_mask, scale_params=True
    )
    ddp_policy = DomainDistrParamPolicy(prior=prior, **ddp_policy_hparam)
    subsubrtn_distr_hparam = dict(
        max_iter=5,
        pop_size=200,
        num_rollouts=1,
        num_is_samples=10,
        expl_std_init=1e-2,
        expl_std_min=1e-5,
        extra_expl_std_init=1e-2,
        extra_expl_decay_iter=5,
        num_workers=num_workers,
    )
    subsubrtn_distr = CEM(ex_dir, env_sim, ddp_policy, **subsubrtn_distr_hparam)
    subrtn_distr_hparam = dict(
        metric=None,
        obs_dim_weight=[1, 1, 1, 1, 10, 10],
        num_rollouts_per_distr=len(dp_map)*10,
        num_workers=num_workers,
    )
    subrtn_distr = SysIdViaEpisodicRL(subsubrtn_distr, behavior_policy=behav_policy, **subrtn_distr_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=15,
        num_eval_rollouts=5,
        warmstart=True,
        thold_succ_subrtn=100,
        subrtn_snapshot_mode='latest',
    )
    algo = SimOpt(ex_dir, env_sim, env_real, subrtn_policy, subrtn_distr, **algo_hparam)

    # Save the environments and the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=args.seed),
        dict(behav_policy=behav_policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(ddp_policy=ddp_policy_hparam, ddp_policy_name=ddp_policy.name),
        dict(subrtn_distr=subrtn_distr_hparam, subrtn_distr_name=subrtn_distr.name),
        dict(subsubrtn_distr=subsubrtn_distr_hparam, subsubrtn_distr_name=subsubrtn_distr.name),
        dict(subrtn_policy=subrtn_policy_hparam, subrtn_policy_name=subrtn_policy.name),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=args.seed)
