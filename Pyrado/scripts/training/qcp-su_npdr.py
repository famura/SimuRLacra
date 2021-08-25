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
Domain parameter identification experiment on the Quanser Cart-Pole environment
using Neural Posterior Domain Randomization
"""
import os.path as osp

import sbi.utils as sbiutils
import torch as to
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.environment_specific import QCartPoleSwingUpAndBalanceCtrl, QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.sbi_embeddings import (
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    DynamicTimeWarpingEmbedding,
    RNNEmbedding,
)
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

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        QCartPoleSwingUpSim.name,
        f"{NPDR.name}_{QQubeSwingUpAndBalanceCtrl.name}",
        num_segs_str + len_seg_str + seed_str,
    )
    t_end = 7  # s

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(dt=1 / 250.0, max_steps=int(t_end * 250))
    env_sim = QCartPoleSwingUpSim(**env_sim_hparams)

    # Create the ground truth target domain and the behavioral policy
    env_real = osp.join(pyrado.EVAL_DIR, f"qcp-su_ectrl_250Hz_{t_end}s_filt")
    policy = QCartPoleSwingUpAndBalanceCtrl(env_sim.spec)  # replaced by the recorded actions if use_rec_act=True

    # Define a mapping: index - domain parameter
    dp_mapping = {0: "voltage_thold_neg", 1: "voltage_thold_pos"}
    # dp_mapping = {
    #     0: "motor_efficiency",
    #     1: "gear_efficiency",
    #     2: "pole_damping",
    #     3: "combined_damping",
    #     4: "cart_friction_coeff",
    #     5: "gear_ratio",
    #     6: "motor_back_emf",
    #     7: "pole_mass",
    #     8: "pole_length",
    # }
    # gravity_const=9.81,  # gravity constant [m/s**2]
    # m_cart=0.38,  # mass of the cart [kg]
    # l_rail=0.814,  # length of the rail the cart is running on [m]
    # eta_m=0.9,  # motor efficiency [-], default 1.
    # eta_g=0.9,  # planetary gearbox efficiency [-], default 1.
    # K_g=3.71,  # planetary gearbox gear ratio [-]
    # J_m=3.9e-7,  # rotor inertia [kg*m**2]
    # r_mp=6.35e-3,  # motor pinion radius [m]
    # R_m=2.6,  # motor armature resistance [Ohm]
    # k_m=7.67e-3,  # motor torque constant [N*m/A] = back-EMF constant [V*s/rad]
    # B_pole=0.0024,  # viscous coefficient at the pole [N*s]
    # B_eq=5.4,  # equivalent Viscous damping coefficient [N*s/m]
    # pole_mass=m_pole,  # mass of the pole [kg]
    # pole_length=l_pole,  # half pole length [m]
    # mu_cart=0.02,  # Coulomb friction coefficient cart-rail [-]

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        low=to.tensor([-1.0, 0.0]),
        # low=to.tensor(
        #     [
        #         0.6,
        #         0.6,
        #         dp_nom["pole_damping"] * 0.5,
        #         dp_nom["combined_damping"] * 0.5,
        #         0,
        #         dp_nom["gear_ratio"] * 0.8,
        #         dp_nom["motor_back_emf"] * 0.8,
        #         dp_nom["pole_mass"] * 0.9,
        #         dp_nom["pole_length"] * 0.9,
        #     ]
        # ),
        high=to.tensor([0.0, 1.0])
        # high=to.tensor(
        #     [
        #         1,
        #         1,
        #         dp_nom["pole_damping"] * 1.5,
        #         dp_nom["combined_damping"] * 1.5,
        #         dp_nom["cart_friction_coeff"] * 5,
        #         dp_nom["gear_ratio"] * 1.2,
        #         dp_nom["motor_back_emf"] * 1.2,
        #         dp_nom["pole_mass"] * 1.1,
        #         dp_nom["pole_length"] * 1.1,
        #     ]
        # ),
    )
    prior = sbiutils.BoxUniform(**prior_hparam)

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
    embedding = create_embedding(BayesSimEmbedding.name, env_sim.spec, **embedding_hparam)

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=5)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=1,
        num_sim_per_round=2000,
        num_sbi_rounds=2,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=10,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        stop_on_done=False,
        use_rec_act=use_rec_act,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=20,  # default: 10
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
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=False),
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
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
