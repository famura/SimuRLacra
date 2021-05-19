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
Domain parameter identification experiment on the Quanser Qube environment using Neural Posterior Domain Randomization
"""
from copy import deepcopy

import torch as to
from sbi import utils
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.sbi_embeddings import (
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    DynamicTimeWarpingEmbedding,
    LastStepEmbedding,
    RNNEmbedding,
)
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.utils.argparser import get_argparser
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{NPDR.name}_{QQubeSwingUpAndBalanceCtrl.name}", "sim2sim")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 250.0, max_steps=1500)
    env_sim = QQubeSwingUpSim(**env_sim_hparams)
    env_sim = ActDelayWrapper(env_sim)

    # Create a fake ground truth target domain
    num_real_rollouts = 2
    env_real = deepcopy(env_sim)
    dp_nom = env_sim.get_nominal_domain_param()
    env_real.domain_param = dict(
        Dr=dp_nom["Dr"] * 1.9,
        Dp=dp_nom["Dp"] * 0.4,
        Rm=dp_nom["Rm"] * 1.0,
        km=dp_nom["km"] * 1.0,
        Mp=dp_nom["Mp"] * 1.1,
        Mr=dp_nom["Mr"] * 1.2,
        Lp=dp_nom["Lp"] * 0.8,
        Lr=dp_nom["Lr"] * 0.9,
        g=dp_nom["g"] * 1.0,
    )
    # randomizer = DomainRandomizer(
    #     NormalDomainParam(name="Dr", mean=dp_nom["Dr"] * 2.0, std=dp_nom["km"] / 10, clip_lo=0.0),
    #     NormalDomainParam(name="Dp", mean=dp_nom["Dp"] * 2.0, std=dp_nom["km"] / 10, clip_lo=0.0),
    #     NormalDomainParam(name="Rm", mean=dp_nom["Rm"] * 1.1, std=dp_nom["km"] / 50, clip_lo=0.0),
    #     NormalDomainParam(name="Km", mean=dp_nom["km"] * 0.9, std=dp_nom["km"] / 50, clip_lo=0.0),
    # )
    # env_real = DomainRandWrapperBuffer(env_real, randomizer)
    # env_real.fill_buffer(num_real_rollouts)

    # Behavioral policy
    policy_hparam = dict(energy_gain=0.587, ref_energy=0.827)
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec, **policy_hparam)

    # Define a mapping: index - domain parameter
    # dp_mapping = {0: "act_delay"}
    # dp_mapping = {0: "Mr", 1: "Mp", 2: "Lr", 3: "Lp"}
    dp_mapping = {0: "Dr", 1: "Dp", 2: "Rm", 3: "km", 4: "Mr", 5: "Mp", 6: "Lr", 7: "Lp", 8: "g"}

    # Prior and Posterior (normalizing flow)
    prior_hparam = dict(
        # low=to.tensor([0.0]),
        # high=to.tensor([5.0]),
        low=to.tensor(
            [
                dp_nom["Dr"] * 0,
                dp_nom["Dp"] * 0,
                dp_nom["Rm"] * 0.8,
                dp_nom["km"] * 0.8,
                dp_nom["Mr"] * 0.8,
                dp_nom["Mp"] * 0.8,
                dp_nom["Lr"] * 0.8,
                dp_nom["Lp"] * 0.8,
                dp_nom["g"] * 0.9,
            ]
        ),
        high=to.tensor(
            [
                2 * 0.0015,
                2 * 0.0005,
                dp_nom["Rm"] * 1.2,
                dp_nom["km"] * 1.2,
                dp_nom["Mr"] * 1.2,
                dp_nom["Mp"] * 1.2,
                dp_nom["Lr"] * 1.2,
                dp_nom["Lp"] * 1.2,
                dp_nom["g"] * 1.1,
            ]
        ),
    )
    prior = utils.BoxUniform(**prior_hparam)

    # Time series embedding
    embedding_hparam = dict(
        downsampling_factor=20,
        # len_rollouts=env_sim.max_steps,
        # recurrent_network_type=nn.RNN,
        # only_last_output=True,
        # hidden_size=20,
        # num_recurrent_layers=1,
        # output_size=1,
    )
    embedding = create_embedding(DeltaStepsEmbedding.name, env_sim.spec, **embedding_hparam)

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=5)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=1000,
        num_sbi_rounds=5,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=10,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=False,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=True),
        num_workers=20,
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
        dict(env=env_sim_hparams, seed=args.seed),
        dict(policy=policy_hparam, policy_name=policy.name),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
