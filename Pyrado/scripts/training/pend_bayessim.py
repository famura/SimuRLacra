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
from copy import deepcopy

import torch as to
from sbi import utils

import pyrado
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.playback import PlaybackPolicy
from pyrado.policies.special.environment_specific import create_pend_excitation_policy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, f"{BayesSim.name}_{PlaybackPolicy.name}", "sin")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 50.0, max_steps=400)
    env_sim = PendulumSim(**env_hparams)
    env_sim.domain_param = dict(d_pole=0)
    env_sim.domain_param = dict(tau_max=4.5)

    # Create a fake ground truth target domain
    num_real_rollouts = 1
    env_real = deepcopy(env_sim)

    # Define a mapping: index - domain parameter
    dp_mapping = {0: "m_pole", 1: "l_pole"}

    # Prior
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        low=to.tensor([dp_nom["m_pole"] * 0.3, dp_nom["l_pole"] * 0.3]),
        high=to.tensor([dp_nom["m_pole"] * 1.7, dp_nom["l_pole"] * 1.7]),
    )
    prior = utils.BoxUniform(**prior_hparam)
    # prior_hparam = dict(
    #     loc=to.tensor([dp_nom["m_pole"], dp_nom["l_pole"]]),
    #     covariance_matrix=to.tensor([[dp_nom["m_pole"] / 20, 0], [0, dp_nom["l_pole"] / 20]]),
    # )
    # prior = to.distributions.MultivariateNormal(**prior_hparam)

    # Behavioral policy
    policy = create_pend_excitation_policy(env_sim, 1)

    # Algorithm
    algo_hparam = dict(
        num_components=5,
        component_perturbation=1e-2,
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=200,
        num_sbi_rounds=3,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=1000,
        downsampling_factor=1,
        num_segments=1,
        subrtn_sbi_training_hparam=dict(
            training_batch_size=50,  # default: 50
            learning_rate=5e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        subrtn_policy=None,
        num_workers=20,
    )
    algo = BayesSim(
        save_dir=ex_dir,
        env_sim=env_sim,
        env_real=env_real,
        policy=policy,
        dp_mapping=dp_mapping,
        prior=prior,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy_name=policy.name),
        dict(prior=prior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
