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
Script to identify the domain parameters of the Pendulum environment using Neural Posterior Domain Randomization
"""
import copy
import math

import numpy as np
import torch as to
import torch.nn as nn
from sbi import utils
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.domain_randomization.transformations import LogDomainParamTransform
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.dummy import IdlePolicy
from pyrado.sampling.sbi_embeddings import (
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    DynamicTimeWarpingEmbedding,
    LastStepEmbedding,
    RNNEmbedding,
)
from pyrado.utils.argparser import get_argparser
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, f"{NPDR.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=400)
    env_sim = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))

    # Create a fake ground truth target domain
    num_real_rollouts = 1
    env_real = copy.deepcopy(env_sim)
    # randomizer = DomainRandomizer(
    #     NormalDomainParam(name="m", mean=0.8, std=0.8 / 50),
    #     NormalDomainParam(name="k", mean=33.0, std=33 / 50),
    #     NormalDomainParam(name="d", mean=0.3, std=0.3 / 50),
    # )
    # env_real = DomainRandWrapperBuffer(env_real, randomizer)
    # env_real.fill_buffer(num_real_rollouts)
    env_real.domain_param = dict(m=0.8, k=40, d=0.3)

    # Behavioral policy
    policy = IdlePolicy(env_sim.spec)

    # Define a mapping: index - domain parameter
    env_sim = LogDomainParamTransform(env_sim, ["m"])
    dp_mapping = {0: "m", 1: "k", 2: "d"}

    # Prior
    dp_nom = env_sim.get_nominal_domain_param()  # m=1.0, k=30.0, d=0.5
    prior_hparam = dict(
        low=to.tensor([math.log(dp_nom["m"] * 0.5), dp_nom["k"] * 0.5, dp_nom["d"] * 0.5]),
        high=to.tensor([math.log(dp_nom["m"] * 1.5), dp_nom["k"] * 1.5, dp_nom["d"] * 1.5]),
    )
    prior = utils.BoxUniform(**prior_hparam)

    # Time series embedding
    embedding_hparam = dict(
        downsampling_factor=1,
        # len_rollouts=env_sim.max_steps,
        # recurrent_network_type=nn.RNN,
        # only_last_output=True,
        # hidden_size=20,
        # num_recurrent_layers=1,
        # output_size=1,
    )
    embedding = create_embedding(DynamicTimeWarpingEmbedding.name, env_sim.spec, **embedding_hparam)

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=30, num_transforms=4)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=400,
        num_sbi_rounds=2,
        simulation_batch_size=4,
        normalize_posterior=False,
        num_eval_samples=2,
        num_segments=2,
        # len_segments=50,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=30,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=False),
        num_workers=4,
    )
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        embedding,
        subrtn_sbi_class=SNPE_C,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
