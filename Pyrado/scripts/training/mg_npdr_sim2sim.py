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
Domain parameter identification experiment on the Mini-Golf environment using Neural Posterior Domain Randomization
"""
import copy
import math

import sbi.utils as sbiutils
import torch as to
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environments.rcspysim.mini_golf import MiniGolfIKSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.poly_time import PolySplineTimePolicy
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
    ex_dir = setup_experiment(MiniGolfIKSim.name, f"{NPDR.name}_{PolySplineTimePolicy.name}", "sim2sim")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(
        relativeZdTask=True,
        physicsEngine="Bullet",
        dt=0.01,
        max_steps=int(15 / 0.01),
        checkJointLimits=True,
        fixedInitState=False,
        observeForceTorque=False,
    )
    env_sim = MiniGolfIKSim(**env_sim_hparams)

    # Create a fake ground truth target domain
    num_real_rollouts = 1
    env_real = copy.deepcopy(env_sim)
    dp_nom = env_sim.get_nominal_domain_param()
    env_real.domain_param = dict(
        ball_radius=dp_nom["ball_radius"] * 1.2,
        ball_mass=dp_nom["ball_mass"] * 0.9,
        club_mass=dp_nom["club_mass"] * 1.2,
        ball_slip=dp_nom["ball_slip"] * 1.0,
        ball_friction_coefficient=dp_nom["ball_friction_coefficient"] * 1.0,
        ball_rolling_friction_coefficient=dp_nom["ball_rolling_friction_coefficient"] * 0.01,
        ground_friction_coefficient=dp_nom["ground_friction_coefficient"] * 1.5,
        obstacleleft_pos_offset_x=-0.0,
        obstacleleft_pos_offset_y=-0.1,
        obstacleleft_rot_offset_c=15 / 180 * math.pi,
        obstacleright_pos_offset_x=0.0,
        obstacleright_pos_offset_y=0.1,
        obstacleright_rot_offset_c=-15 / 180 * math.pi,
    )
    env_real.reset()  # need to call reset for RcsPySim envs to actually create a new simulation with the new params

    # Behavioral policy
    policy_hparam = dict(
        t_end=1.0,
        cond_lvl="vel",
        # Zd (rel), Y (rel), Zdist (abs), PHI (abs), THETA (abs)
        cond_final=[
            [0.0, 0.0, 0.01, math.pi / 2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        cond_init=[
            [-7.0, 0.0, 0.01, math.pi / 2, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        overtime_behavior="hold",
    )
    policy = PolySplineTimePolicy(env_sim.spec, env_sim.dt, **policy_hparam)

    # Define a mapping: index - domain parameter
    dp_mapping = {
        0: "ball_radius",
        1: "ball_mass",
        2: "club_mass",
        3: "ball_friction_coefficient",
        4: "ball_rolling_friction_coefficient",
        5: "ball_slip",
        # X: "ground_slip",
        6: "ground_friction_coefficient",
        7: "obstacleleft_pos_offset_x",
        8: "obstacleleft_pos_offset_y",
        9: "obstacleleft_rot_offset_c",
        10: "obstacleright_pos_offset_x",
        11: "obstacleright_pos_offset_y",
        12: "obstacleright_rot_offset_c",
    }

    # Prior and Posterior (normalizing flow)
    prior_hparam = dict(
        # low=to.tensor([0.0]),
        # high=to.tensor([5.0]),
        low=to.tensor(
            [
                dp_nom["ball_radius"] * 0.5,
                dp_nom["ball_mass"] * 0.5,
                dp_nom["club_mass"] * 0.5,
                dp_nom["ball_friction_coefficient"] * 0.0,
                0,  # ball_rolling_friction_coefficient
                0,  # ball_slip
                dp_nom["ground_friction_coefficient"] * 0.0,
                -0.2,  # obstacleleft_pos_offset_x
                -0.2,  # obstacleleft_pos_offset_y
                -math.pi / 2,  # obstacleleft_rot_offset_c
                -0.2,  # obstacleright_pos_offset_x
                -0.2,  # obstacleright_pos_offset_y
                -math.pi / 2,  # obstacleright_rot_offset_c
            ]
        ),
        high=to.tensor(
            [
                dp_nom["ball_radius"] * 1.5,
                dp_nom["ball_mass"] * 1.5,
                dp_nom["club_mass"] * 1.5,
                dp_nom["ball_friction_coefficient"] * 2.0,
                1e-4,  # ball_rolling_friction_coefficient
                1e-3,  # ball_slip
                dp_nom["ground_friction_coefficient"] * 2.0,
                0.2,  # obstacleleft_pos_offset_x
                0.2,  # obstacleleft_pos_offset_y
                math.pi / 2,  # obstacleleft_rot_offset_c
                0.2,  # obstacleright_pos_offset_x
                0.2,  # obstacleright_pos_offset_y
                math.pi / 2,  # obstacleright_rot_offset_c
            ]
        ),
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
        state_mask_labels=("Ball_X", "Ball_Y"),
    )
    embedding = create_embedding(BayesSimEmbedding.name, env_sim.spec, **embedding_hparam)

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=8)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=1000,
        num_sbi_rounds=3,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=10,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        stop_on_done=False,
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
        save_dir=ex_dir,
        env_sim=env_sim,
        env_real=env_real,
        policy=policy,
        dp_mapping=dp_mapping,
        prior=prior,
        embedding=embedding,
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
