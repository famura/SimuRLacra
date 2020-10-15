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
Train an agent to solve the WAM Ball-in-cup environment using Bayesian Domain Randomization.
Continue in the same directory of a previous experiment.
"""
import joblib
import os.path as osp
import torch as to

import pyrado
from pyrado.algorithms.power import PoWER
from pyrado.algorithms.bayrn import BayRn
from pyrado.logger.experiment import load_dict_from_yaml, ask_for_experiment
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Environments
    hparams = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
    env_sim = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))
    env_real = joblib.load(osp.join(ex_dir, 'env_real.pkl'))

    # Policy
    policy = to.load(osp.join(ex_dir, 'policy.pt'))

    # Subroutine
    subroutine_hparam = hparams['subroutine']
    power = PoWER(ex_dir, env_sim, policy, **subroutine_hparam)

    # Set the boundaries for the GP
    bounds = to.load(osp.join(ex_dir, 'bounds.pt'))

    # Algorithm
    algo = BayRn(ex_dir, env_sim, env_real, subrtn=power, bounds=bounds, **hparams['algo'])

    # Jeeeha
    seed = hparams['seed'] if hparams['seed'] != 'None' else None
    algo.train(snapshot_mode='best', seed=seed, load_dir=ex_dir)
