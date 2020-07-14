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
Script to test the bi-manual box lifting task using a hard-coded time-based policy
"""
import rcsenv
import pyrado
from pyrado.environments.rcspysim.box_lifting import BoxLiftingVelMPsSim, BoxLiftingPosMPsSim
from pyrado.policies.dummy import IdlePolicy
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(0)


def create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxLiftingVelMPsSim(
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={'file': 'collisionModel.xml'},
        checkJointLimits=checkJointLimits,
    )
    env.reset(domain_param=env.get_nominal_domain_param())

    # Set up policy
    policy = IdlePolicy(env.spec)  # don't move at all

    return env, policy


def create_position_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        # return [1, 0, 1, 1, 1,
        #         0, 0, 0, 0, 0, 1]
        return [0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 1]

    # Set up environment
    env = BoxLiftingPosMPsSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        fixed_init_state=True,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={'file': 'collisionModel.xml'},
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
        observeDSGoalDistance=True,
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_velocity_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        return [1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1]

    # Set up environment
    env = BoxLiftingVelMPsSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        fixed_init_state=True,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={'file': 'collisionModel.xml'},
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
        observeDSGoalDistance=False,
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


if __name__ == '__main__':
    # Choose setup
    setup_type = 'vel'  # idle, pos, vel
    physicsEngine = 'Bullet'  # Bullet or Vortex
    graphFileName = 'gBoxLifting_posCtrl.xml'  # gBoxLifting_trqCtrl or gBoxLifting_posCtrl
    dt = 1/100.
    max_steps = int(20/dt)
    ref_frame = 'basket'  # world, box, basket, or table
    checkJointLimits = False

    if setup_type == 'idle':
        env, policy = create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits)
    elif setup_type == 'pos':
        env, policy = create_position_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame,
                                                checkJointLimits)
    elif setup_type == 'vel':
        env, policy = create_velocity_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame,
                                                checkJointLimits)
    else:
        raise pyrado.ValueErr(given=setup_type, eq_constraint="'idle', 'pos', 'vel")

    # Simulate and plot
    print('observations:\n', env.obs_space.labels)
    done, param, state = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), eval=True, max_steps=max_steps,
                     reset_kwargs=dict(domain_param=param, init_state=state), stop_on_done=False)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
