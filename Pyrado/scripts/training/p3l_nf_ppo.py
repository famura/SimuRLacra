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
Train an agent to solve the Planar-3-Link task using Neural Fields and Proximal Policy Optimization.
"""
import torch as to

from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.ppo import PPO
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environments.rcspysim.planar_3_link import Planar3LinkIKActivationSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.fnn import FNNPolicy, FNN
from pyrado.policies.neural_fields import NFPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(Planar3LinkIKActivationSim.name, PPO.name, f'{FNNPolicy.name}_obsnorm', seed=101)

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        dt=1/50.,
        max_steps=1200,
        task_args=dict(consider_velocities=True),
        max_dist_force=None,
        checkJointLimits=True,
        collisionAvoidanceIK=True,
        observeVelocities=True,
        observeForceTorque=True,
        observeCollisionCost=False,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeDynamicalSystemGoalDistance=True,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
    )
    env = Planar3LinkIKActivationSim(**env_hparams)
    env = ActNormWrapper(env)
    eub = {
        'GD_DS0': 2.,
        'GD_DS1': 2.,
        'GD_DS2': 2.,
    }
    env = ObsNormWrapper(env, explicit_ub=eub)
    # env = ObsNormWrapper(env)
    # env = ObsPartialWrapper(env, idcs=['Effector_Xd', 'Effector_Zd'])
    env = ObsPartialWrapper(env, idcs=['Effector_DiscrepTS_X', 'Effector_DiscrepTS_Z'])
    # env = ObsPartialWrapper(env, idcs=['Effector_DiscrepTS_X', 'Effector_DiscrepTS_Z', 'Effector_Xd', 'Effector_Zd'])

    # Policy
    policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    policy = FNNPolicy(spec=env.spec, **policy_hparam)
    print(policy)

    # Critic
    value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.999,
        lamda=0.95,
        num_epoch=10,
        batch_size=512,
        standardize_adv=False,
        standardizer=None,
        # max_grad_norm=5.,
        lr=1e-3,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        min_steps=20*env.max_steps,
        num_epoch=5,
        eps_clip=0.15,
        batch_size=512,
        std_init=0.6,
        # max_grad_norm=5.,
        lr=5e-4,
        num_sampler_envs=6,
    )
    algo = PPO(ex_dir, env, policy, critic, **algo_hparam)

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml([
        dict(env=env_hparams, seed=ex_dir.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name)],
        ex_dir
    )

    # Jeeeha
    algo.train(seed=ex_dir.seed, snapshot_mode='best')
