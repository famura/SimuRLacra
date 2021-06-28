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
import os.path as osp

import torch as to
from sbi.inference import SNPE_C
from torch.distributions import MultivariateNormal, Normal

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.domain_randomization.transformations import SqrtDomainParamTransform
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.dummy import DummyPolicy
from pyrado.policies.feed_forward.time import TimePolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding, DeltaStepsEmbedding, RNNEmbedding
from pyrado.utils.argparser import get_argparser
from pyrado.utils.sbi import create_embedding


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

    use_rec_act = True
    ectl = True

    # Experiment (set seed before creating the modules)
    if ectl:
        ex_dir = setup_experiment(
            QQubeSwingUpSim.name,
            f"{NPDR.name}_{QQubeSwingUpAndBalanceCtrl.name}",
            num_segs_str + len_seg_str + seed_str,
        )
        t_end = 5  # s
    else:
        ex_dir = setup_experiment(
            QQubeSwingUpSim.name,
            f"{NPDR.name}_{TimePolicy.name}",
            num_segs_str + len_seg_str + seed_str,
        )
        t_end = 10  # s

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 250.0, max_steps=int(t_end * 250))
    env_sim = QQubeSwingUpSim(**env_sim_hparams)
    # env_sim = ActDelayWrapper(env_sim)

    # Create the ground truth target domain and the behavioral policy
    if ectl:
        env_real = osp.join(pyrado.EVAL_DIR, f"qq-su_ectrl_250Hz_{t_end}s")  # 5s long
        policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)  # replaced by the recorded actions if use_rec_act=True
    else:
        env_real = osp.join(pyrado.EVAL_DIR, f"qq_chrip_10to0Hz_+1.5V_250Hz_{t_end}s")
        assert use_rec_act
        policy = DummyPolicy(env_sim.spec)  # replaced by recorded real actions

    # Define a mapping: index - domain parameter
    dp_mapping = {
        0: "damping_rot_pole",
        1: "damping_pend_pole",
        2: "motor_resistance",
        3: "motor_back_emf",
        4: "mass_rot_pole",
        5: "mass_pend_pole",
        6: "length_rot_pole",
        7: "length_pend_pole",
        8: "gravity_const",
    }

    # Transform the domain parameter space
    env_sim = SqrtDomainParamTransform(env_sim, [dp_name for dp_name in dp_mapping.values()])

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        loc=env_sim.forward(
            to.tensor(
                [
                    dp_nom["damping_rot_pole"],
                    dp_nom["damping_pend_pole"],
                    dp_nom["motor_resistance"],
                    dp_nom["motor_back_emf"],
                    dp_nom["mass_rot_pole"],
                    dp_nom["mass_pend_pole"],
                    dp_nom["length_rot_pole"],
                    dp_nom["length_pend_pole"],
                    dp_nom["gravity_const"],
                ]
            )
        ),
        covariance_matrix=to.diagflat(
            env_sim.forward(
                env_sim.forward(
                    to.tensor(
                        [
                            dp_nom["damping_rot_pole"] / 10,
                            dp_nom["damping_pend_pole"] / 5,
                            dp_nom["motor_resistance"] / 5,
                            dp_nom["motor_back_emf"] / 10,
                            dp_nom["mass_rot_pole"] / 10,
                            dp_nom["mass_pend_pole"] / 10,
                            dp_nom["length_rot_pole"] / 20,
                            dp_nom["length_pend_pole"] / 20,
                            dp_nom["gravity_const"] / 20,
                        ]
                    ),
                )
            )
        ),
    )
    prior = MultivariateNormal(**prior_hparam)

    # Time series embedding
    embedding_hparam = dict(
        downsampling_factor=1,
        # len_rollouts=env_sim.max_steps,
    )
    embedding = create_embedding(BayesSimEmbedding.name, env_sim.spec, **embedding_hparam)

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=5)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=2,
        num_sim_per_round=1000,
        num_sbi_rounds=4,
        simulation_batch_size=1,
        normalize_posterior=False,
        num_eval_samples=2,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        stop_on_done=True,
        use_rec_act=use_rec_act,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=True),
        num_workers=12,
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
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
