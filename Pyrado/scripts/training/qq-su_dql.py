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
Train an agent to solve the Quanser Qube swing-up task using Proximal Policy Optimization.
"""
import pyrado
import torch as to
from pyrado.algorithms.step_based.dql import DQL
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.environment_wrappers.action_discrete import ActDiscreteWrapper
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.feed_forward.fnn import FNNPolicy, FNN, DiscreteActQValPolicy
from pyrado.utils.argparser import get_argparser
from torch.optim import lr_scheduler

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QQubeSwingUpSim.name, f"{DQL.name}_{DiscreteActQValPolicy.name}", f"100Hz_seed_{args.seed}"
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 100.0, max_steps=500)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActDiscreteWrapper(env)

    # Policy
    net = FNN(
        input_size=DiscreteActQValPolicy.get_qfcn_input_size(env.spec),
        hidden_sizes=[64, 64],
        hidden_nonlin=to.tanh,
        output_size=DiscreteActQValPolicy.get_qfcn_output_size(),
    )
    policy = DiscreteActQValPolicy(spec=env.spec, net=net)

    algo_hparam = dict(
        memory_size=50000,
        eps_init=1.0,
        eps_schedule_gamma=0.9,
        gamma=0.99,
        max_iter=5000,
        num_updates_per_step=None,
        target_update_intvl=500,
        num_init_memory_steps=1000,
        min_rollouts=None,
        min_steps=10,
        batch_size=32,
        eval_intvl=100,
        max_grad_norm=0.5,
        lr=0.0005,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
        num_workers=1,
    )
    algo = DQL(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed), dict(algo=algo_hparam, algo_name=algo.name), save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="latest", seed=args.seed)
