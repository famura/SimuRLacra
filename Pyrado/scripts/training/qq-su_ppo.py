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
Train an agent to solve the Qube swing-up task using Proximal Policy Optimization.
"""
import torch as to

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.spaces import ValueFunctionSpace
from pyrado.algorithms.ppo import PPO
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.rnn import RNNPolicy, LSTMPolicy, GRUPolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f'{PPO.name}_{FNNPolicy.name}', '100Hz')

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=600)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    init_param_kwargs = dict(uniform_bias=True)
    policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh, init_param_kwargs=init_param_kwargs)
    # policy_hparam = dict(hidden_size=32, num_recurrent_layers=1)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)
    # policy = RNNPolicy(spec=env.spec, **policy_hparam)
    # policy = LSTMPolicy(spec=env.spec, **policy_hparam)
    # policy = GRUPolicy(spec=env.spec, **policy_hparam)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh, init_param_kwargs=init_param_kwargs)
    # value_fcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = RNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = LSTMPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    # value_fcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
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

    # Subroutine
    algo_hparam = dict(
        max_iter=600,
        min_steps=23*env.max_steps,
        num_epoch=7,
        eps_clip=0.0744,
        batch_size=60,
        std_init=0.9074,
        lr=3.446e-04,
        max_grad_norm=1.,
        num_workers=16,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, value_fcn=value_fcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=args.seed)
