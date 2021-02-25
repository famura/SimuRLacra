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
Domain parameter identification experiment on the Pendulum environment using Neural Posterior Domain Randomization
"""
import numpy as np
import torch as to
import torch.nn as nn
from copy import deepcopy
from sbi.inference import SNPE
from sbi import utils

import pyrado
from pyrado.algorithms.inference.lfi import NPDR
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.time import PlaybackPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, f"{NPDR.name}", "sin")

    # Set seed if desired
    pyrado.set_seed(0, verbose=True)
    # pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=800)
    env_sim = PendulumSim(**env_hparams)

    # Create a fake ground truth target domain
    env_real = deepcopy(env_sim)
    # env_real.domain_param = dict(m_pole=1 / 1.25 ** 2, l_pole=1.25)
    env_real.domain_param = dict(m_pole=1.0, l_pole=1.0)

    # Define a mapping: index - domain parameter
    dp_mapping = {0: "m_pole", 1: "l_pole"}

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        low=to.tensor([dp_nom["m_pole"] * 0.2, dp_nom["l_pole"] * 0.2]),
        high=to.tensor([dp_nom["m_pole"] * 1.8, dp_nom["l_pole"] * 1.8]),
    )
    prior = utils.BoxUniform(**prior_hparam)
    posterior_nn_hparam = dict(model="maf", embedding_net=nn.Identity(), hidden_features=100, num_transforms=10)

    # Policy
    policy_hparam = dict(amp_max=dp_nom["tau_max"], f_sin=0.5)

    def fcn_of_time(t: float):
        act = policy_hparam["amp_max"] * np.sin(2 * np.pi * t * policy_hparam["f_sin"]) + np.random.randn(1) / 50
        return act.repeat(env_sim.act_space.flat_dim)

    num_real_rollouts = 1
    act_recordings = [
        [fcn_of_time(t) for t in np.arange(0, env_sim.max_steps * env_sim.dt, env_sim.dt)]
        for _ in range(num_real_rollouts)
    ]
    policy = PlaybackPolicy(env_sim.spec, act_recordings)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        summary_statistic="bayessim",  # bayessim or dtw_distance
        num_real_rollouts=num_real_rollouts,
        num_sbi_rounds=3,
        num_sim_per_round=500,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=100,
        num_segments=1,
        # len_segments=40,
        sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=30,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
        ),
        num_workers=4,
    )
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        posterior_nn_hparam,
        SNPE,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(posterior_nn=posterior_nn_hparam),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
