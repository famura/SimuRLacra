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
Continue a SimOpt experiment in the same folder
"""
import joblib
import os.path as osp
import torch as to

from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.power import PoWER
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.reps import REPS
from pyrado.algorithms.simopt import SimOpt
from pyrado.algorithms.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.environments.quanser.quanser_qube import QQubeReal
from pyrado.logger.experiment import load_dict_from_yaml, ask_for_experiment
from pyrado.policies.domain_distribution import DomainDistrParamPolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import wrap_like_other_env


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    num_workers = 16

    # Environments
    hparams = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
    env_sim = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))
    env_real = joblib.load(osp.join(ex_dir, 'env_real.pkl'))
    # env_sim.dt = 1/500
    # env_sim.max_steps = 3000
    # env_real = QQubeReal(dt=1/500, max_steps=3000, ip='192.168.2.40')

    # Wrap the real environment in the same way as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Subroutine for policy improvement
    behav_policy = to.load(osp.join(ex_dir, 'policy.pt'))
    subrtn_policy_hparams = hparams['subrtn_policy']
    subrtn_policy_hparams.update({'num_workers': num_workers})
    if hparams['subrtn_policy_name'] == PoWER.name:
        subrtn_policy = PoWER(ex_dir, env_sim, behav_policy, **subrtn_policy_hparams)
    elif hparams['subrtn_policy_name'] == PPO.name:
        critic_hparam = hparams.get('critic', dict(
            gamma=0.9995,
            lamda=0.9648,
            num_epoch=2,
            batch_size=500,
            standardize_adv=False,
            lr=5.792e-4,
            max_grad_norm=1.,
        ))
        value_fcn = to.load(osp.join(ex_dir, 'valuefcn.pt'))
        critic = GAE(value_fcn, **critic_hparam)

        subrtn_policy = PPO(ex_dir, env_sim, behav_policy, critic, **subrtn_policy_hparams)
    else:
        raise NotImplementedError

    # Subroutine for system identification
    try:
        ddp_policy = to.load(osp.join(ex_dir, 'ddp_policy.pt'))
    except FileNotFoundError:
        prior = joblib.load(osp.join(ex_dir, 'prior.pkl'))
        ddp_policy = DomainDistrParamPolicy(prior=prior, **hparams['ddp_policy'])
    subsubrtn_distr_hparams = hparams['subsubrtn_distr']
    subsubrtn_distr_hparams.update({'num_workers': num_workers})
    if hparams['subsubrtn_distr_name'] == CEM.name:
        subsubrtn_distr = CEM(ex_dir, env_sim, ddp_policy, **subsubrtn_distr_hparams)
    elif hparams['subsubrtn_distr_name'] == REPS.name:
        subsubrtn_distr = REPS(ex_dir, env_sim, ddp_policy, **subsubrtn_distr_hparams)
    else:
        raise NotImplementedError
    subrtn_distr_hparams = hparams['subrtn_distr']
    subrtn_distr = SysIdViaEpisodicRL(subsubrtn_distr, behavior_policy=behav_policy, **subrtn_distr_hparams)

    # Algorithm
    algo_hparam = hparams['algo']
    algo = SimOpt(ex_dir, env_sim, env_real, subrtn_policy, subrtn_distr, **algo_hparam)

    # Jeeeha
    seed = hparams['seed'] if isinstance(hparams['seed'], int) else None
    algo.train(snapshot_mode='latest', seed=seed, load_dir=ex_dir)
