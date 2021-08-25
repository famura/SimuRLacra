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
Domain parameter identification experiment on the joint space controlled WAM environment using Neural Posterior
Domain Randomization
"""
import sbi.utils as sbiutils
import torch as to
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environments.mujoco.wam_jsc import WAMJointSpaceCtrlSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.time import TimePolicy
from pyrado.policies.special.environment_specific import wam_jsp_7dof_sin
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding
from pyrado.utils.argparser import get_argparser
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(WAMJointSpaceCtrlSim.name, f"{NPDR.name}", "sim2sim")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(num_dof=7, dt=1 / 250.0, max_steps=10 * 250)
    env_sim = WAMJointSpaceCtrlSim(**env_sim_hparams)

    # Create a fake ground truth target domain
    num_real_rollouts = 3
    env_real = WAMJointSpaceCtrlSim(**env_sim_hparams)
    dp_nom = env_sim.get_nominal_domain_param()
    env_real.domain_param = dict(
        link_1_mass=dp_nom["link_1_mass"] * 1.3,
        link_2_mass=dp_nom["link_2_mass"] * 1.3,
        link_3_mass=dp_nom["link_3_mass"] * 1.3,
        link_4_mass=dp_nom["link_4_mass"] * 1.3,
        link_5_mass=dp_nom["link_5_mass"] * 1.3,
        link_6_mass=dp_nom["link_6_mass"] * 1.3,
        link_7_mass=dp_nom["link_7_mass"] * 1.3,
        # joint_1_damping=dp_nom["joint_1_damping"] * 10,
        # joint_2_damping=dp_nom["joint_2_damping"] * 10,
        # joint_3_damping=dp_nom["joint_3_damping"] * 10,
        # joint_4_damping=dp_nom["joint_4_damping"] * 10,
        # joint_5_damping=dp_nom["joint_5_damping"] * 10,
        # joint_6_damping=dp_nom["joint_6_damping"] * 10,
        # joint_7_damping=dp_nom["joint_7_damping"] * 10,
        # joint_1_dryfriction=dp_nom["joint_1_dryfriction"] * 10,
        # joint_2_dryfriction=dp_nom["joint_2_dryfriction"] * 10,
        # joint_3_dryfriction=dp_nom["joint_3_dryfriction"] * 10,
        # joint_4_dryfriction=dp_nom["joint_4_dryfriction"] * 10,
        # joint_5_dryfriction=dp_nom["joint_5_dryfriction"] * 10,
        # joint_6_dryfriction=dp_nom["joint_6_dryfriction"] * 10,
        # joint_7_dryfriction=dp_nom["joint_7_dryfriction"] * 10,
    )

    # Define a mapping: index - domain parameter
    dp_mapping = {
        0: "link_1_mass",
        1: "link_2_mass",
        2: "link_3_mass",
        3: "link_4_mass",
        4: "link_5_mass",
        5: "link_6_mass",
        6: "link_7_mass",
    }
    # dp_mapping = {
    #     0: "joint_1_damping",
    #     1: "joint_2_damping",
    #     2: "joint_3_damping",
    #     3: "joint_4_damping",
    #     4: "joint_5_damping",
    #     5: "joint_6_damping",
    #     6: "joint_7_damping",
    # }
    # dp_mapping = {
    #     0: "joint_1_dryfriction",
    #     1: "joint_2_dryfriction",
    #     2: "joint_3_dryfriction",
    #     3: "joint_4_dryfriction",
    #     4: "joint_5_dryfriction",
    #     5: "joint_6_dryfriction",
    #     6: "joint_7_dryfriction",
    # }
    # dp_mapping = create_damping_dryfriction_domain_param_map_wamjsc()

    # Behavioral policy
    policy_hparam = dict()
    policy = TimePolicy(env_real.spec, wam_jsp_7dof_sin, env_real.dt)

    # Prior
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        low=to.tensor([dp_nom[name] * 0.5 for name in dp_mapping.values()]),
        high=to.tensor([dp_nom[name] * 1.5 for name in dp_mapping.values()]),
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
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=1000,
        num_sbi_rounds=3,
        simulation_batch_size=20,
        normalize_posterior=False,
        num_eval_samples=10,
        # num_segments=5,
        len_segments=100,
        stop_on_done=False,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=5e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
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
        dict(posterior_nn=posterior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="latest", seed=args.seed)
