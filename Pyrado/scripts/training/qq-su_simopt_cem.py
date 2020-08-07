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
Train an agent to solve the Quanser Qube environment using Simulation Optimization running Cross-Entropy Method.
"""
import torch as to

from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.simopt import SimOpt
from pyrado.algorithms.sysid_as_rl import SysIdByEpisodicRL, DomainDistrParamPolicy
from pyrado.domain_randomization.domain_parameter import UniformDomainParam, NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper, DomainRandWrapperLive
from pyrado.environments.pysim.quanser_qube import QQubeStabSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeStabSim.name, f'{SimOpt.name}-{CEM.name}', seed=1001)
    num_workers = 6

    # Environments
    env_hparams = dict(dt=1/100., max_steps=500)
    env_real = QQubeStabSim(**env_hparams)
    env_real.domain_param = dict(
        Rm=8.4*0.8,
    )

    env_sim = QQubeStabSim(**env_hparams)
    randomizer = DomainRandomizer(
        NormalDomainParam(name='R_m', mean=0, std=1e-12),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {0: ('R_m', 'mean'), 1: ('R_m', 'std')}
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Subroutine for policy improvement
    behav_policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    behav_policy = FNNPolicy(spec=env_sim.spec, **behav_policy_hparam)
    value_fcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)  # FNN
    value_fcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.9885,
        lamda=0.9648,
        num_epoch=2,
        batch_size=60,
        standardize_adv=False,
        lr=5.792e-4,
        max_grad_norm=1.,
    )
    critic = GAE(value_fcn, **critic_hparam)
    subrtn_policy_hparam = dict(
        max_iter=100,
        min_steps=23*env_sim.max_steps,
        num_workers=num_workers,
        num_epoch=7,
        eps_clip=0.0744,
        batch_size=60,
        std_init=0.9074,
        lr=3.446e-04,
        max_grad_norm=1.,
    )
    subrtn_policy = PPO(ex_dir, env_sim, behav_policy, critic, **subrtn_policy_hparam)

    # Subroutine for system identification
    prior = DomainRandomizer(
        NormalDomainParam(name='R_m', mean=8.4, std=8.4e-2),
    )
    ddp_policy = DomainDistrParamPolicy(mapping=dp_map, prior=prior)
    subsubrtn_distr_hparam = dict(
        max_iter=100,
        pop_size=40,
        num_rollouts=1,
        num_is_samples=4,
        expl_std_init=0.1,
        expl_std_min=0.001,
        extra_expl_std_init=0.,
        extra_expl_decay_iter=10,
        num_workers=num_workers,
    )
    subsubrtn = CEM(ex_dir, env_sim, ddp_policy, **subsubrtn_distr_hparam)
    subrtn_hparam = dict(
        metric=None,
        obs_dim_weight=[1., 1., 1., 1.],
        num_rollouts_per_distr=50,
        num_workers=num_workers,
    )
    subrtn_distr = SysIdByEpisodicRL(subsubrtn, behavior_policy=behav_policy, **subrtn_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=200,
    )
    algo = SimOpt(ex_dir, env_sim, env_real, subrtn_policy, subrtn_distr, **algo_hparam)

    # Save the environments and the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(behav_policy=behav_policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(subsubrtn_distr=subsubrtn_distr_hparam, subsubrtn_distr_name=CEM.name),
        dict(subrtn_distr=subrtn_hparam, subrtn_distr_name=SysIdByEpisodicRL.name, dp_map=dp_map),
        dict(subrtn_policy=subrtn_policy_hparam, subrtn_policy_name=CEM.name),
        dict(algo=algo_hparam, algo_name=SimOpt.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
