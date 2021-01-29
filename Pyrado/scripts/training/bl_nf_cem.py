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
from pyrado.algorithms.episodic.cem import CEM
from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environments.rcspysim.box_lifting import BoxLiftingVelIKActivationSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.recurrent.neural_fields import NFPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BoxLiftingVelIKActivationSim.name, f"{CEM.name}_{NFPolicy.name}", "obsnoise")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(
        physicsEngine="Bullet",
        graphFileName="gBoxLifting_trqCtrl.xml",
        dt=0.01,
        max_steps=150,
        ref_frame="basket",
        collisionConfig={"file": "collisionModel.xml"},
        fixedInitState=True,
        checkJointLimits=True,
        taskCombinationMethod="sum",
        collisionAvoidanceIK=False,
        observeVelocity=False,
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
    env = ObsPartialWrapper(env, idcs=["Box_Y", "Box_Z", "Box_A"])
    env = GaussianObsNoiseWrapper(env, noise_std=[0.01, 0.01, 2.0, 2.0])

    # Domain randomizer
    dp_nom = env.get_nominal_domain_param()
    randomizer = create_default_randomizer(env)
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        hidden_size=21,
        conv_out_channels=1,
        mirrored_conv_weights=True,
        conv_kernel_size=1,
        conv_padding_mode="circular",
        init_param_kwargs=dict(bell=True),
        activation_nonlin=to.tanh,
        tau_init=50.0,
        tau_learnable=True,
        kappa_init=0,
        kappa_learnable=False,
        potential_init_learnable=False,
    )
    policy = NFPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=50,
        pop_size=100,
        num_init_states_per_domain=4,
        num_domains=1,
        num_is_samples=40,
        expl_std_init=1.0,
        expl_std_min=0.02,
        extra_expl_std_init=0.5,
        extra_expl_decay_iter=5,
        full_cov=False,
        symm_sampling=False,
        num_workers=4,
    )
    algo = CEM(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
