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
Domain parameter identification experiment on the Mini-Golf environment using BayesSim
"""
import math
import os.path as osp

import sbi.utils as sbiutils
import torch as to

import pyrado
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.environments.rcspysim.mini_golf import MiniGolfJointCtrlSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.time import TimePolicy
from pyrado.policies.special.environment_specific import create_mg_joint_pos_policy
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding
from pyrado.utils.argparser import get_argparser
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(MiniGolfJointCtrlSim.name, f"{BayesSim.name}_{TimePolicy.name}", "")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim_hparams = dict(
        relativeZdTask=True,
        physicsEngine="Bullet",
        dt=0.01,
        max_steps=int(8 / 0.01),
        checkJointLimits=True,
        fixedInitState=True,
        observeForceTorque=False,
    )
    env_sim = MiniGolfJointCtrlSim(**env_sim_hparams)

    # Set up environment
    env_real = MiniGolfJointCtrlSim(
        physicsEngine="Bullet",
        dt=0.01,
        max_steps=int(8 / 0.01),
        checkJointLimits=True,
        fixedInitState=True,
        graphFileName="gMiniGolf_gt.xml",
        physicsConfigFile="pMiniGolf_gt.xml",
    )
    # env_real = osp.join(pyrado.EVAL_DIR, "mg-jnt_100Hz_8s_filt")

    # Behavioral policy
    policy_hparam = dict(t_strike_end=0.5)
    policy = create_mg_joint_pos_policy(env_sim, **policy_hparam)

    # Define a mapping: index - domain parameter
    # dp_mapping = {
    #     0: "ball_radius",
    #     1: "ball_mass",
    #     2: "ball_slip",
    #     3: "ball_friction_coefficient",
    #     4: "ball_rolling_friction_coefficient",
    #     5: "club_mass",
    #     6: "ground_slip",
    #     7: "ground_friction_coefficient",
    #     8: "obstacleleft_pos_offset_x",
    #     9: "obstacleleft_pos_offset_y",
    #     10: "obstacleleft_rot_offset_c",
    #     11: "obstacleright_pos_offset_x",
    #     12: "obstacleright_pos_offset_y",
    #     13: "obstacleright_rot_offset_c",
    # }
    # dp_mapping = {
    #     0: "ball_radius",
    #     1: "ball_mass",
    #     2: "ball_restitution",
    #     # 3: "ball_friction_coefficient",
    #     3: "ball_rolling_friction_coefficient",
    #     # 4: "obstacleleft_pos_offset_x",
    #     # 5: "obstacleleft_pos_offset_y",
    #     # 6: "obstacleleft_rot_offset_c",
    #     4: "obstacleright_pos_offset_x",
    #     5: "obstacleright_pos_offset_y",
    #     6: "obstacleright_rot_offset_c",
    # }
    dp_mapping = {
        0: "ball_radius",
        1: "ball_mass",
        2: "ball_restitution",
        3: "ball_rolling_friction_coefficient",
        4: "obstacleleft_pos_offset_x",
        5: "obstacleleft_pos_offset_y",
        6: "obstacleleft_rot_offset_c",
        7: "obstacleright_pos_offset_x",
        8: "obstacleright_pos_offset_y",
        9: "obstacleright_rot_offset_c",
    }

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        low=to.tensor(
            [
                0.014,  # ball_radius
                0.0025,  # ball_mass
                0,  # ball_restitution
                # 0,  # ball_friction_coefficient
                0,  # ball_rolling_friction_coefficient
                # dp_nom["club_mass"] * 0.2,
                # dp_nom["ground_slip"] * 0,
                # dp_nom["ground_friction_coefficient"] * 0,
                -0.08,  # obstacleleft_pos_offset_x
                -0.08,  # obstacleleft_pos_offset_y
                -math.pi,  # obstacleleft_rot_offset_c
                -0.08,  # obstacleright_pos_offset_x
                -0.08,  # obstacleright_pos_offset_y
                -math.pi,  # obstacleright_rot_offset_c
            ]
        ),
        high=to.tensor(
            [
                0.026,  # ball_radius
                0.0075,  # ball_mass
                1.0,  # ball_restitution
                # 2.0,  # ball_friction_coefficient
                0.0005,  # ball_rolling_friction_coefficient
                # dp_nom["club_mass"] * 1.8,
                # dp_nom["ground_slip"] * 2,
                # dp_nom["ground_friction_coefficient"] * 2,
                0.08,  # obstacleleft_pos_offset_x
                0.08,  # obstacleleft_pos_offset_y
                math.pi,  # obstacleleft_rot_offset_c
                0.08,  # obstacleright_pos_offset_x
                0.08,  # obstacleright_pos_offset_y
                math.pi,  # obstacleright_rot_offset_c
            ]
        ),
    )
    prior = sbiutils.BoxUniform(**prior_hparam)

    # Time series embedding
    embedding_hparam = dict(
        downsampling_factor=2,
        state_mask_labels=("Ball_X", "Ball_Y"),
        # act_mask_labels=("Z Velocity [m/s]",),
    )
    embedding = create_embedding(BayesSimEmbedding.name, env_sim.spec, **embedding_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_components=20,
        num_real_rollouts=2,
        num_sim_per_round=50000,
        num_sbi_rounds=1,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=10,
        num_segments=1,
        stop_on_done=False,
        use_rec_act=True,
        subrtn_sbi_training_hparam=dict(
            training_batch_size=50,  # default: 50
            learning_rate=3e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        num_workers=20,
    )
    algo = BayesSim(
        save_dir=ex_dir,
        env_sim=env_sim,
        env_real=env_real,
        policy=policy,
        dp_mapping=dp_mapping,
        prior=prior,
        embedding=embedding,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_sim_hparams, seed=args.seed),
        dict(dp_mapping=dp_mapping),
        dict(policy=policy_hparam, policy_name=policy.name),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
