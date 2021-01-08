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
Train an agent to solve the box-lifting task using Activation Dynamics Networks and Hill Climbing.
"""
import torch as to

import pyrado
from pyrado.algorithms.episodic.hc import HCNormal
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.domain_randomization.domain_parameter import NormalDomainParam, UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.rcspysim.box_lifting import BoxLiftingVelIKActivationSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.recurrent.adn import pd_cubic, ADNPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxLiftingVelIKActivationSim.name, f"{HCNormal.name}_{ADNPolicy.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(
        physicsEngine="Bullet",
        graphFileName="gBoxLifting_posCtrl.xml",
        dt=0.01,
        max_steps=1200,
        ref_frame="basket",
        collisionConfig={"file": "collisionModel.xml"},
        fixedInitState=True,
        checkJointLimits=True,
        taskCombinationMethod="sum",
        collisionAvoidanceIK=False,
        observeVelocity=True,
        observeCollisionCost=False,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=False,
        observeForceTorque=True,
        observeDynamicalSystemGoalDistance=False,
    )
    env = BoxLiftingVelIKActivationSim(**env_hparams)

    # randomizer = create_default_randomizer(env)
    dp_nom = env.get_nominal_domain_param()
    randomizer = DomainRandomizer(
        NormalDomainParam(name="box_length", mean=dp_nom["box_length"], std=dp_nom["box_length"], clip_lo=5e-2),
        NormalDomainParam(name="box_width", mean=dp_nom["box_width"], std=dp_nom["box_width"], clip_lo=5e-2),
        NormalDomainParam(name="box_mass", mean=dp_nom["box_mass"], std=dp_nom["box_mass"] / 5),
        UniformDomainParam(
            name="box_friction_coefficient",
            mean=dp_nom["box_friction_coefficient"],
            halfspan=dp_nom["box_friction_coefficient"] / 5,
            clip_lo=1e-5,
        ),
        NormalDomainParam(name="basket_mass", mean=dp_nom["basket_mass"], std=dp_nom["basket_mass"] / 5),
        UniformDomainParam(
            name="basket_friction_coefficient",
            mean=dp_nom["basket_friction_coefficient"],
            halfspan=dp_nom["basket_friction_coefficient"] / 5,
            clip_lo=1e-5,
        ),
    )
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        tau_init=50.0,
        tau_learnable=True,
        kappa_init=1e-3,
        kappa_learnable=True,
        activation_nonlin=to.sigmoid,
        potentials_dyn_fcn=pd_cubic,
        potential_init_learnable=False,
    )
    policy = ADNPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=5 * policy.num_param,
        num_rollouts=1,
        expl_factor=1.05,
        expl_std_init=1.0,
        num_workers=6,
    )
    algo = HCNormal(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml(
        [
            dict(env=env_hparams, seed=args.seed),
            dict(policy=policy_hparam),
            dict(algo=algo_hparam, algo_name=algo.name),
        ],
        ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
