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
Script to test the bi-manual ball-in-tube task using a hard-coded time-based policy
"""
import rcsenv
import pyrado
from pyrado.environments.rcspysim.ball_in_tube import BallInTubeVelDSSim, BallInTubePosDSSim, BallInTubeIKActivationSim, \
    BallInTubeIKSim
from pyrado.policies.dummy import IdlePolicy
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(4)


def create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BallInTubeVelDSSim(
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        checkJointLimits=checkJointLimits,
    )
    env.reset(domain_param=env.get_nominal_domain_param())

    # Set up policy
    policy = IdlePolicy(env.spec)  # don't move at all

    return env, policy


def create_ik_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        if t < 2:
            return [0.001, 0.001, 0.001, 0.002, 0.002, 0.002,
                    0.001, 0.001, 0.001, 0.002, 0.002, 0.002]
        else:
            return [0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0]

    # Set up environment
    env = BallInTubeIKSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        fixed_init_state=True,
        ref_frame=ref_frame,
        checkJointLimits=checkJointLimits,
        collisionAvoidanceIK=True,
        observeVelocity=False,
        observeCollisionCost=True,
        observePredictedCollisionCost=True,
        observeManipulabilityIndex=True,
        observeCurrentManipulability=True,
        observeTaskSpaceDiscrepancy=True,
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_ik_activation_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        if t < 3:
            return [0, 0, 0.1,
                    0, 0, 0, 0.1]
        elif t < 5:
            return [0, 0, 0.001,
                    0, 0.3, 0, 0.001]
        elif t < 7:
            return [0, 0, 0.001,
                    0, 0, 1, 0.001]
        elif t < 12:
            return [0, 1, 0.001,
                    0, 0, 0, 0.001]
        elif t < 15:
            return [1, 0, 0,
                    1, 0, 0, 0]
        else:
            return [0, 0, 0,
                    0, 0, 0, 0]

    # Set up environment
    env = BallInTubeIKActivationSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        fixed_init_state=True,
        ref_frame=ref_frame,
        taskCombinationMethod='sum',
        checkJointLimits=checkJointLimits,
        collisionAvoidanceIK=True,
        observeVelocity=False,
        observeCollisionCost=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=True,
        observeCurrentManipulability=True,
        observeTaskSpaceDiscrepancy=True,
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_position_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        return [0.2, 0,
                0.5, 0]

    # Set up environment
    env = BallInTubePosDSSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        fixed_init_state=True,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        taskCombinationMethod='sum',
        checkJointLimits=checkJointLimits,
        collisionAvoidanceIK=False,
        observeVelocity=False,
        observeCollisionCost=True,
        observePredictedCollisionCost=True,
        observeManipulabilityIndex=True,
        observeCurrentManipulability=True,
        observeDynamicalSystemDiscrepancy=True,
        observeTaskSpaceDiscrepancy=True,
        observeDynamicalSystemGoalDistance=True,
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_velocity_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        return [1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1]

    # Set up environment
    env = BallInTubeVelDSSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        fixed_init_state=True,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        taskCombinationMethod='sum',
        checkJointLimits=checkJointLimits,
        collisionAvoidanceIK=False,
        observeVelocity=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=False,
        observeManipulabilityIndex=False,
        observeCurrentManipulability=True,
        observeDynamicalSystemDiscrepancy=False,
        observeTaskSpaceDiscrepancy=True,
        observeForceTorque=True,
        observeDynamicalSystemGoalDistance=False,
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


if __name__ == '__main__':
    # Choose setup
    setup_type = 'ik_activation'  # idle, ik, ik_activation, ds_activation_pos, ds_activation_vel
    common_hparam = dict(
        physicsEngine='Bullet',  # Bullet or Vortex
        graphFileName='gBallInTube_trqCtrl.xml',  # gBallInTube_trqCtrl
        dt=1/100.,
        max_steps=int(20*100),
        ref_frame='table',  # world, table, or slider
        checkJointLimits=False,
    )
    if setup_type == 'idle':
        env, policy = create_idle_setup(**common_hparam)
    elif setup_type == 'ik':
        env, policy = create_ik_setup(**common_hparam)
    elif setup_type == 'ik_activation':
        env, policy = create_ik_activation_setup(**common_hparam)
    elif setup_type == 'ds_activation_pos':
        env, policy = create_position_mps_setup(**common_hparam)
    elif setup_type == 'ds_activation_vel':
        env, policy = create_velocity_mps_setup(**common_hparam)
    else:
        raise pyrado.ValueErr(given=setup_type, eq_constraint="'idle', 'ds_activation_pos', 'ds_activation_vel")

    # Simulate and plot
    print('observations:\n', env.obs_space.labels)
    done, param, state = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), stop_on_done=False,
                     eval=True, max_steps=common_hparam['max_steps'],
                     reset_kwargs=dict(domain_param=param, init_state=state))
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
