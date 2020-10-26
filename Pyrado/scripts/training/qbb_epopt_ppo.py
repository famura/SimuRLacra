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
Train an agent to solve the Quanser Ball-Balancer environment using Ensemble Policy Optimization.
"""
from numpy import pi

from pyrado.algorithms.meta.epopt import EPOpt
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.algorithms.step_based.gae import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.domain_randomization.default_randomizers import create_default_randomizer_qbb
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import *
from pyrado.policies.fnn import FNNPolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QBallBalancerSim.name, f'{EPOpt.name}-{PPO.name}_{FNNPolicy.name}',
                              'obsnoise_actnorm_actdelay-30')

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1/500., max_steps=2500)
    env = QBallBalancerSim(**env_hparams)
    env = GaussianObsNoiseWrapper(env, noise_std=[1/180*pi, 1/180*pi, 0.0025, 0.0025,  # [rad, rad, m, m, ...
                                                  2/180*pi, 2/180*pi, 0.05, 0.05])  # ... rad/s, rad/s, m/s, m/s]
    env = ActNormWrapper(env)
    env = ActDelayWrapper(env)
    randomizer = create_default_randomizer_qbb()
    randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=15, halfspan=15, clip_lo=0, roundint=True))
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)  # FNN
    # policy_hparam = dict(hidden_size=64, num_recurrent_layers=1)  # LSTM & GRU
    policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = RNNPolicy(spec=env.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    # value_fcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.9995,
        lamda=0.98,
        num_epoch=5,
        batch_size=100,
        lr=5e-4,
        standardize_adv=False,
        # max_grad_norm=1.,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Subroutine
    subrtn_hparam = dict(
        max_iter=1000,
        min_steps=30*env.max_steps,
        num_workers=4,
        num_epoch=5,
        eps_clip=0.1,
        batch_size=100,
        std_init=0.8,
        lr=2e-4,
        # max_grad_norm=1.,
    )
    subrtn = PPO(ex_dir, env, policy, critic, **subrtn_hparam)

    # Meta-Algorithm
    algo_hparam = dict(skip_iter=100, epsilon=0.2, gamma=critic.gamma)
    algo = EPOpt(env, subrtn, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(subrtn=subrtn_hparam, subrtn_name=subrtn.name),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=args.seed)
