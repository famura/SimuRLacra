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
Train an agent to solve the PlanarInsert task using Activation Dynamics Networks and Natural Evolution Strategies.
"""
import torch as to

import pyrado
from pyrado.algorithms.episodic.nes import NES
from pyrado.domain_randomization.default_randomizers import create_empty_randomizer
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environments.rcspysim.planar_insert import PlanarInsertTASim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.recurrent.adn import ADNPolicy, pd_cubic
from pyrado.policies.feed_forward.fnn import FNN
from pyrado.utils.argparser import get_argparser


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PlanarInsertTASim.name, f'{NES.name}_{ADNPolicy.name}', 'obsnorm_actdelay-4')

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(
        physicsEngine='Vortex',  # Bullet or Vortex
        graphFileName='gPlanarInsert6Link.xml',
        dt=1/50.,
        max_steps=800,
        # max_dist_force=1e1,
        taskCombinationMethod='sum',  # 'sum', 'mean',  'product', or 'softmax'
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeForceTorque=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeDynamicalSystemGoalDistance=False,
        observeDynamicalSystemDiscrepancy=True,
        observeTaskSpaceDiscrepancy=True,
        usePhysicsNode=True,
    )
    env = PlanarInsertTASim(**env_hparams)
    # Explicit normalization bounds
    elb = {
        'DiscrepDS_Effector_X': -1.,
        'DiscrepDS_Effector_Z': -1.,
        'DiscrepDS_Effector_Bd': -1,
    }
    eub = {
        'DiscrepDS_Effector_X': 1.,
        'DiscrepDS_Effector_Z': 1.,
        'DiscrepDS_Effector_Bd': 1,
    }
    env = ObsNormWrapper(env, explicit_lb=elb, explicit_ub=eub)

    randomizer = create_empty_randomizer()
    env = ActDelayWrapper(env)
    randomizer.add_domain_params(UniformDomainParam(name='act_delay', mean=2, halfspan=2, clip_lo=0, roundint=True))
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(
        obs_layer=FNN(input_size=env.obs_space.flat_dim,
                      output_size=env.act_space.flat_dim,
                      hidden_sizes=[32, 32],
                      hidden_nonlin=to.tanh,
                      dropout=0.),
        tau_init=5.,
        tau_learnable=True,
        kappa_init=0.02,
        kappa_learnable=True,
        activation_nonlin=to.sigmoid,
        potentials_dyn_fcn=pd_cubic,
    )
    policy = ADNPolicy(spec=env.spec, dt=env.dt, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=5000,
        pop_size=None,
        num_rollouts=1,
        eta_mean=1.,
        eta_std=None,
        expl_std_init=1.0,
        symm_sampling=False,
        transform_returns=True,
        num_workers=12,
    )
    algo = NES(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(snapshot_mode='latest', seed=args.seed)
