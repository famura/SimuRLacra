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
Script to test the simplified box flipping task using a hard-coded time-based policy
"""
import rcsenv
import pyrado
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.rcspysim.box_flipping import BoxFlippingVelDSSim, BoxFlippingPosDSSim, \
    BoxFlippingIKActivationSim
from pyrado.policies.dummy import IdlePolicy
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(1)


def create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxFlippingVelDSSim(
        usePhysicsNode=True,
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

    # Set up policy
    policy = IdlePolicy(env.spec)  # don't move at all

    return env, policy


def create_ik_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxFlippingIKActivationSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        ref_frame=ref_frame,
        collisionConfig={'file': 'collisionModel.xml'},
        checkJointLimits=checkJointLimits,
    )

    # Set up policy
    def policy(t: float):
        # return [0.1, 0.05, 0.1,  # left Y, Z, dist
        #         -0.1, 0.05, 0.1]  # right X, Y, dist
        return [0.0, 0.05,  # left Y, Z
                -0.1, 0.05]  # right X, Y

    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_position_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxFlippingPosDSSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={'file': 'collisionModel.xml'},
        checkJointLimits=checkJointLimits,
        collisionAvoidanceIK=False,
        observeVelocities=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=True,
        observeManipulabilityIndex=True,
        observeCurrentManipulability=True,
        observeDynamicalSystemDiscrepancy=True,
        observeTaskSpaceDiscrepancy=False,
        observeDynamicalSystemGoalDistance=True,
    )

    # Set up policy
    # def policy(t: float):
    #     if t < 3.1:
    #         return [0, 0, 0, 0,
    #                 0, 0, 0, 0, 1]
    #     elif t <= 4.5:
    #         return [0, 0, 0, 0,
    #                 0, 0, 0, 0, 1]
    #     else:
    #         return [0, 0, 0, 0,
    #                 0, 0, 0, 0, 1]
    def policy(t: float):
        if t <= 5:
            return [0.2, 0, 0, 0,
                    0, 1]
        else:
            return [-0.1, 0, 0, 0,
                    0.1, 1.]

    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_velocity_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxFlippingVelDSSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={'file': 'collisionModel.xml'},
        checkJointLimits=checkJointLimits,
        collisionAvoidanceIK=False,
        observeVelocities=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=True,
        observeManipulabilityIndex=True,
        observeCurrentManipulability=True,
        observeDynamicalSystemDiscrepancy=True,
        observeTaskSpaceDiscrepancy=True,
    )

    # Set up policy
    def policy(t: float):
        if t < 2.5:
            return [.8, 0., 0., 0.,
                    0, 0, 0, 0]
        elif t <= 3.:
            return [0.2, 0., .8, 0.,
                    0, 0, 0, 0]
        else:
            return [0., 0.15, 0., 0.,
                    0, 0, 0, 0]

    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


if __name__ == '__main__':
    # Choose setup
    setup_type = 'ik_activation'  # idle, ik, pos, or vel
    physicsEngine = 'Bullet'  # Bullet or Vortex
    graphFileName = 'gBoxFlipping_posCtrl.xml'  # gBoxFlipping_posCtrl.xml or gBoxFlipping_trqCtrl.xml
    dt = 1/100.
    max_steps = int(20/dt)
    ref_frame = 'table'  # world, box, table, or table
    checkJointLimits = False
    randomize = False

    if setup_type == 'idle':
        env, policy = create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits)
    elif setup_type == 'ik_activation':
        env, policy = create_ik_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits)
    elif setup_type == 'pos':
        env, policy = create_position_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame,
                                                checkJointLimits)
    elif setup_type == 'vel':
        env, policy = create_velocity_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame,
                                                checkJointLimits)
    else:
        raise pyrado.ValueErr(given=setup_type, eq_constraint="'idle', 'pos', 'vel'")

    if randomize:
        dp_nom = env.get_nominal_domain_param()
        randomizer = DomainRandomizer(
            UniformDomainParam(name='box_mass', mean=dp_nom['box_mass'], halfspan=dp_nom['box_mass']/5),
            UniformDomainParam(name='box_width', mean=dp_nom['box_width'], halfspan=dp_nom['box_length']/5)
        )
        env = DomainRandWrapperLive(env, randomizer)

    # Simulate and plot
    print('observations:\n', env.obs_space.labels)
    done, param, state = False, None, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(text=False, video=True), eval=True, max_steps=max_steps,
                     reset_kwargs=dict(domain_param=param, init_state=state), stop_on_done=False)
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
