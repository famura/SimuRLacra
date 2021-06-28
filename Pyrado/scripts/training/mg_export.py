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
Export a control policy and a pre-strike policy for the Mini-Golf experiment
"""
import math
import os

import rcsenv
import torch as to

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environments.rcspysim.mini_golf import MiniGolfIKSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.features import FeatureStack, const_feat
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.policies.feed_forward.poly_time import PolySplineTimePolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import cpp_export


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(MiniGolfIKSim.name, f"{NPDR.name}_{PolySplineTimePolicy.name}", "export")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    dt = 0.01
    relativeZdTask = True
    env_hparams = dict(
        usePhysicsNode=True,
        physicsEngine="Bullet",
        dt=dt,
        max_steps=int(15 / dt),
        checkJointLimits=True,
        fixedInitState=True,
        collisionAvoidanceIK=False,
        observeForceTorque=False,
        relativeZdTask=relativeZdTask,
    )
    env = MiniGolfIKSim(**env_hparams)

    # Set up the policies
    if relativeZdTask:
        policy_hparam = dict(
            t_end=0.6,
            cond_lvl="vel",
            # Zd (rel), Y (rel), Zdist (abs), PHI (abs), THETA (abs)
            cond_final=[
                [0.0, 0.0, 0.01, math.pi / 2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            cond_init=[
                [-100.0, 0.0, 0.01, math.pi / 2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            overtime_behavior="hold",
        )
    else:
        policy_hparam = dict(
            t_end=3.0,
            cond_lvl="vel",
            # X (abs), Y (rel), Z (abs), A (abs), C (abs)
            # cond_final=[[0.5, 0.0, 0.04, -0.876], [0.5, 0.0, 0.0, 0.0]],
            # cond_init=[[0.1, 0.0, 0.04, -0.876], [0.0, 0.0, 0.0, 0.0]],
            # X (abs), Y (rel), Zdist (abs), PHI (abs), THETA (abs)
            cond_final=[
                [0.9, 0.0, 0.005, math.pi / 2, 0.0],  # math.pi / 2 - 0.4
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            cond_init=[
                [0.3, 0.0, 0.01, math.pi / 2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            overtime_behavior="hold",
        )
    policy = PolySplineTimePolicy(env.spec, dt, **policy_hparam)
    pre_strike_policy_hparam = dict(feats=FeatureStack(const_feat))
    pre_strike_policy = LinearPolicy(env.spec, **pre_strike_policy_hparam)
    x_des = to.tensor(
        # X (abs), Y (rel), Z_dist (abs), C (abs)
        # [0.1, 0.0, 0.04, -0.876]
        # Zd (rel), Y (rel2), Zdist (abs), PHI (abs), THETA (abs)
        [0.0, 0.0, 0.015, 1.05 * math.pi / 2, 0.0]
    )
    pre_strike_policy.init_param(init_values=x_des)

    # Save Python objects
    pyrado.save(env, "env.pkl", ex_dir)
    pyrado.save(policy, "policy.pt", ex_dir, use_state_dict=False)
    pyrado.save(policy, "pre_strike_policy.pt", ex_dir, use_state_dict=False)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam, policy_name=policy.name),
        dict(pre_strike_policy=pre_strike_policy_hparam, pre_strike_policy_name=pre_strike_policy.name),
        save_dir=ex_dir,
    )

    # Export the policies to C++ and the experiment's config
    for save_dir in [ex_dir, os.path.join(rcsenv.RCSPYSIM_CONFIG_PATH, "MiniGolf")]:
        cpp_export(save_dir, policy, env, write_policy_node=True)
        cpp_export(
            save_dir,
            pre_strike_policy,
            env=None,
            policy_export_name="pre_strike_policy_export",
            write_policy_node=True,
            policy_node_name="preStrikePolicy",
        )
