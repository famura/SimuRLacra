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
import torch as to

from pyrado.algorithms.svpg import SVPG
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, SVPG.name, seed=1001)

    # Environment
    env_hparams = dict(dt=1/100., max_steps=500)
    env = BallOnBeamSim(**env_hparams)
    env = ActNormWrapper(env)

    # Specification of actor an critic (will be instantiated in SVPG)
    actor_hparam = dict(
        hidden_sizes=[64],
        hidden_nonlin=to.relu,
    )
    value_fcn_hparam = dict(
        hidden_sizes=[32],
        hidden_nonlin=to.tanh,
    )
    critic_hparam = dict(
        gamma=0.995,
        lamda=0.95,
        num_epoch=5,
        lr=1e-3,
        standardize_adv=False,
        max_grad_norm=5.,
    )
    particle_hparam = dict(actor=actor_hparam, value_fcn=value_fcn_hparam, critic=critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=200,
        min_steps=30*env.max_steps,
        num_particles=3,
        temperature=1,
        lr=1e-3,
        std_init=1.0,
        horizon=50,
        num_workers=12,
    )
    algo = SVPG(ex_dir, env, particle_hparam, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed)
