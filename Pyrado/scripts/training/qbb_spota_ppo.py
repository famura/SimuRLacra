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
Train an agent to solve the Quanser Ball-Balancer environment using
Simulation-based Policy Optimization with Transferability Assessment.
"""
from copy import deepcopy
from numpy import pi

import pyrado
from pyrado.algorithms.step_based.gae import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.algorithms.meta.spota import SPOTA
from pyrado.domain_randomization.default_randomizers import create_conservative_randomizer
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.recurrent.rnn import GRUPolicy
from pyrado.sampling.sequences import *
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QBallBalancerSim.name, f'{SPOTA.name}-{PPO.name}_{GRUPolicy.name}',
                              'actnorm_obsnoise_actedlay-10_ws')

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment and domain randomization
    env_hparams = dict(dt=1/100., max_steps=500, load_experimental_tholds=True)
    env = QBallBalancerSim(**env_hparams)
    env = GaussianObsNoiseWrapper(env, noise_std=[1/180*pi, 1/180*pi, 0.005, 0.005,  # [rad, rad, m, m, ...
                                                  10/180*pi, 10/180*pi, 0.05, 0.05])  # ... rad/s, rad/s, m/s, m/s]
    env = ActNormWrapper(env)
    env = ActDelayWrapper(env)
    randomizer = create_conservative_randomizer(env)
    randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=5, halfspan=5, clip_lo=0, roundint=True))
    env = DomainRandWrapperBuffer(env, randomizer)

    # Policy
    # policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)  # FNN
    # policy_hparam = dict(hidden_size=32, num_recurrent_layers=1, hidden_nonlin='tanh')  # RNN
    policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    # policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = RNNPolicy(spec=env.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    # vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    # vfcn_hparam = dict(hidden_siz=16, num_recurrent_layers=1, hidden_nonlin='tanh')  # RNN
    vfcn_hparam = dict(hidden_size=16, num_recurrent_layers=1)  # LSTM & GRU
    # vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    # vfcn = RNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    # vfcn = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    vfcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.98,
        num_epoch=1,
        batch_size=100,
        lr=1e-4,
        standardize_adv=False,
        max_grad_norm=1.,
    )
    critic_cand = GAE(vfcn, **critic_hparam)
    critic_refs = GAE(deepcopy(vfcn), **critic_hparam)

    subrtn_hparam_cand = dict(
        max_iter=400,
        # min_rollouts=0,  # will be overwritten by SPOTA
        min_steps=0,  # will be overwritten by SPOTA
        num_epoch=1,
        eps_clip=0.1,
        batch_size=100,
        std_init=0.8,
        max_grad_norm=1.,
        lr=1e-4,
    )
    subrtn_hparam_refs = deepcopy(subrtn_hparam_cand)

    sr_cand = PPO(ex_dir, env, policy, critic_cand, **subrtn_hparam_cand)
    sr_refs = PPO(ex_dir, env, deepcopy(policy), critic_refs, **subrtn_hparam_refs)

    # Meta-Algorithm
    algo_hparam = dict(
        max_iter=10,
        alpha=0.05,
        beta=0.1,
        nG=20,
        nJ=180,
        ntau=5,
        nc_init=10,
        nr_init=1,
        sequence_cand=sequence_add_init,
        sequence_refs=sequence_const,
        warmstart_cand=True,
        warmstart_refs=True,
        cand_policy_param_init=None,
        cand_critic_param_init=None,
        num_bs_reps=1000,
        studentized_ci=False,
    )
    algo = SPOTA(ex_dir, env, sr_cand, sr_refs, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(critic_cand_and_ref=critic_hparam),
        dict(subrtn_name=sr_cand.name, subrtn_cand=subrtn_hparam_cand, subrtn_refs=subrtn_hparam_refs),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=args.seed)
