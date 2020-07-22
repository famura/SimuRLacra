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
Learn the domain parameter distribution of masses and lengths of the Quanser Qube while using a handcrafted
randomization for the remaining domain parameters. Continue in the same directory of a previous experiment.
"""
import joblib
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.ppo import PPO
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import load_dict_from_yaml, ask_for_experiment


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = ask_for_experiment()

    # Environments
    hparams = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
    env_sim = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))
    env_real = joblib.load(osp.join(ex_dir, 'env_real.pkl'))

    # Policy
    policy = to.load(osp.join(ex_dir, 'policy.pt'))

    # Critic
    valuefcn = to.load(osp.join(ex_dir, 'valuefcn.pt'))
    critic = GAE(valuefcn, **hparams['critic'])

    # Subroutine
    algo_hparam = hparams['subrtn']
    # algo_hparam.update({'num_sampler_envs': 1})
    ppo = PPO(ex_dir, env_sim, policy, critic, **algo_hparam)

    # Set the boundaries for the GP
    bounds = to.load(osp.join(ex_dir, 'bounds.pt'))

    # Algorithm
    algo = BayRn(ex_dir, env_sim, env_real, subrtn=ppo, bounds=bounds, **hparams['algo'])

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=hparams['seed'], load_dir=ex_dir)
