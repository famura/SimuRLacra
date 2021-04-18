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
import torch as to
from sbi.inference import SNPE_C
from sbi import utils

import pyrado
from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.quanser.quanser_qube import QQubeSwingUpReal
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.sbi_embeddings import (
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    RNNEmbedding,
    DynamicTimeWarpingEmbedding,
)
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    seed_str, num_segs_str, len_seg_str = "", "", ""
    if args.seed is not None:
        seed_str = f"_seed-{args.seed}"
    if args.num_segments is not None:
        num_segs_str = f"numsegs-{args.num_segments}"
    elif args.len_segments is not None:
        len_seg_str = f"lensegs-{args.len_segments}"
    else:
        raise pyrado.ValueErr(msg="Either num_segments or len_segments must not be None, but not both or none!")

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QQubeSwingUpSim.name,
        f"{NPDR.name}_{QQubeSwingUpAndBalanceCtrl.name}",
        num_segs_str + len_seg_str + seed_str,
    )

    t_end = 5.5  # s
    num_workers = 8

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 250.0, max_steps=t_end * 250)
    env_sim = QQubeSwingUpSim(**env_sim_hparams)

    # Create the ground truth target domain and the behavioral policy
    env_real = QQubeSwingUpReal(**env_sim_hparams)
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)

    # Define a mapping: index - domain parameter
    dp_mapping = {
        0: "Dr",
        1: "Dp",
        2: "Rm",
        3: "km",
        4: "Mr",
        5: "Mp",
        6: "Lr",
        7: "Lp",
        8: "g",
        9: "V_thold_neg",
        10: "V_thold_pos",
    }

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        low=to.tensor(
            [
                dp_nom["Dr"] * 0,
                dp_nom["Dp"] * 0,
                dp_nom["Rm"] * 0.6,
                dp_nom["km"] * 0.6,
                dp_nom["Mr"] * 0.7,
                dp_nom["Mp"] * 0.7,
                dp_nom["Lr"] * 0.7,
                dp_nom["Lp"] * 0.7,
                dp_nom["g"] * 0.95,
                -0.15,
                0.0,
            ]
        ),
        high=to.tensor(
            [
                2 * 0.0015,
                2 * 0.0005,
                dp_nom["Rm"] * 1.4,
                dp_nom["km"] * 1.4,
                dp_nom["Mr"] * 1.3,
                dp_nom["Mp"] * 1.3,
                dp_nom["Lr"] * 1.3,
                dp_nom["Lp"] * 1.3,
                dp_nom["g"] * 1.05,
                0,
                0.15,
            ]
        ),
    )
    prior = utils.BoxUniform(**prior_hparam)

    # Time series embedding
    # embedding_hparam = dict()
    # embedding = LastStepEmbedding(env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam)
    # embedding_hparam = dict(downsampling_factor=5)
    # embedding = DeltaStepsEmbedding(
    #     env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), env_sim.max_steps, **embedding_hparam
    # )
    embedding_hparam = dict(downsampling_factor=1)
    embedding = BayesSimEmbedding(env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam)
    # embedding_hparam = dict(downsampling_factor=2)
    # embedding = DynamicTimeWarpingEmbedding(
    #     env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam
    # )
    # embedding_hparam = dict(hidden_size=20, num_recurrent_layers=1, output_size=5, downsampling_factor=1)
    # embedding = RNNEmbedding(
    #     env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), env_sim.max_steps, **embedding_hparam
    # )

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=10)

    # Policy optimization subroutine
    subrtn_policy_hparam = dict(
        max_iter=5,
        pop_size=50,
        num_init_states_per_domain=4,
        num_domains=50,
        num_is_samples=10,
        expl_std_init=2.0,
        expl_std_min=0.02,
        symm_sampling=False,
        num_workers=num_workers,
    )
    subrtn_policy = PoWER(ex_dir, env_sim, policy, **subrtn_policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=4,
        num_sim_per_round=500,
        num_sbi_rounds=5,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=50,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        use_rec_act=True,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=20,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=True),
        num_workers=num_workers,
    )
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        SNPE_C,
        embedding,
        **algo_hparam,
        subrtn_policy=subrtn_policy,
        subrtn_policy_snapshot_mode="best",
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_sim_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
