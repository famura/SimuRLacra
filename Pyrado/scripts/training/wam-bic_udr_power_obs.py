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
Train an agent to solve the WAM Ball-in-cup environment with ball observation/tracking using Policy learning by
Weighting Exploration with the Returns
"""
import numpy as np

import pyrado
from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.meta.udr import UDR
from pyrado.domain_randomization.domain_parameter import NormalDomainParam, UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.mujoco.wam_bic import WAMBallInCupSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.dual_rfb import DualRBFLinearPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment
    ex_dir = setup_experiment(
        WAMBallInCupSim.name, f"{UDR.name}-{PoWER.name}_{DualRBFLinearPolicy.name}", "rand-cs-rl-bm-jd-js"
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(
        num_dof=4,
        max_steps=1500,
        fixed_init_state=False,
        observe_ball=True,
        task_args=dict(
            final_factor=500,
            success_bonus=250,
            Q=np.diag([0.5, 1e-4, 4e1]),
            R=np.diag([0, 0, 1e-1, 2e-1]),
            Q_dev=np.diag([0.0, 0.0, 5]),
            # R_dev=np.diag([0., 0., 1e-3, 1e-3])
        ),
    )
    env = WAMBallInCupSim(**env_hparams)

    # Randomizer
    randomizer = DomainRandomizer(
        UniformDomainParam(name="cup_scale", mean=1.0, halfspan=0.2),
        NormalDomainParam(name="rope_length", mean=0.3, std=0.005),
        NormalDomainParam(name="ball_mass", mean=0.021, std=0.001),
        UniformDomainParam(name="joint_2_damping", mean=0.05, halfspan=0.05),
        UniformDomainParam(name="joint_2_dryfriction", mean=0.1, halfspan=0.1),
    )
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    bounds = ([0.0, 0.25, 0.5], [1.0, 1.5, 2.5])
    policy_hparam = dict(rbf_hparam=dict(num_feat_per_dim=9, bounds=bounds, scale=None), dim_mask=2)
    policy = DualRBFLinearPolicy(env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=15,
        pop_size=100,
        num_is_samples=10,
        num_init_states_per_domain=2,
        num_domains=10,
        expl_std_init=np.pi / 12,
        expl_std_min=0.02,
        num_workers=8,
    )
    algo = PoWER(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed, snapshot_mode="best")
