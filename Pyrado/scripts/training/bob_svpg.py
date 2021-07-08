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
Train agents to solve the Ball-on-Beam environment using Stein Variational Policy Gradient.
"""
from pyrado.logger.step import StepLogger
import torch as to

import pyrado
from pyrado.algorithms.step_based.a2c import A2C
from pyrado.algorithms.step_based.gae import GAE, ValueFunctionSpace
from pyrado.algorithms.step_based.svpg import SVPG
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, SVPG.name)

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparam = dict(dt=1 / 100.0, max_steps=500)
    env = BallOnBeamSim(**env_hparam)
    env = ActNormWrapper(env)

    # Specification of actor an critic (will be instantiated in SVPG)
    actor_hparam = dict(
        hidden_sizes=[64],
        hidden_nonlin=to.relu,
    )
    vfcn_hparam = dict(
        hidden_sizes=[32],
        hidden_nonlin=to.tanh,
    )
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=5,
        lr=1e-3,
        standardize_adv=False,
        max_grad_norm=5.0,
    )

    a2c_hparam = dict(
        max_iter=200,
        min_steps=2*env.max_steps,
        lr=1e-3,

    )

    # Algorithm
    algo_hparam = dict(
        max_iter=200,
        num_particles=3,
        temperature=1,
        horizon=50,
    )

    actor = FNNPolicy(spec=env.spec, **actor_hparam)
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic = GAE(vfcn, **critic_hparam)
    particle_logger = StepLogger()
    particle_example = A2C(ex_dir, env, actor, critic, logger=particle_logger, **a2c_hparam)

    algo = SVPG(ex_dir, env, particle_example, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparam, seed=args.seed),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
