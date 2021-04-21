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
from pyrado.environments.rcspysim.box_lifting import (
    BoxLiftingPosDSSim,
    BoxLiftingPosIKActivationSim,
    BoxLiftingVelDSSim,
    BoxLiftingVelIKActivationSim,
)
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(5)


def create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxLiftingVelDSSim(
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        tasks_left=None,  # use defaults
        tasks_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={"file": "collisionModel.xml"},
        fixedInitState=True,
        taskCombinationMethod="sum",
        checkJointLimits=checkJointLimits,
    )
    env.reset(domain_param=env.get_nominal_domain_param())

    # Set up policy
    policy = IdlePolicy(env.spec)  # don't move at all

    return env, policy


def create_pos_ika_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxLiftingPosIKActivationSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        ref_frame=ref_frame,
        collisionConfig={"file": "collisionModel.xml"},
        fixedInitState=True,
        checkJointLimits=checkJointLimits,
        taskCombinationMethod="sum",
        collisionAvoidanceIK=True,
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
    def policy(t: float):
        if t < 2:
            return [1.0, 1.0, 0.1]  # right Y, Z, dist_box
        elif t < 7:
            return [0.1, 0.1, 1.0]  # right Y, Z, dist_box
        else:
            return [0.0, 0.0, 0.1]  # right Y, Z, dist_box

    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_vel_ika_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxLiftingVelIKActivationSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        ref_frame=ref_frame,
        collisionConfig={"file": "collisionModel.xml"},
        fixedInitState=True,
        checkJointLimits=checkJointLimits,
        taskCombinationMethod="sum",
        collisionAvoidanceIK=True,
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
    def policy(t: float):
        if t < 2:
            return [0.0, -0.2]
        if t < 5.0:
            return [0.3, -0.05]
        if t < 8:
            return [0.15, 0.3]
        elif t < 10:
            return [0.1, 0.45]
        else:
            return [0.0, 0.0]

    policy = TimePolicy(env.spec, policy, dt)

    return env, policy


def create_pos_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        return [1, 0, 1, 0, 0]

    # Set up environment
    env = BoxLiftingPosDSSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        tasks_left=None,  # use defaults
        tasks_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={"file": "collisionModel.xml"},
        fixedInitState=True,
        taskCombinationMethod="sum",
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


def create_vel_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy(t: float):
        if t < 2:
            return [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        elif t < 6:
            return [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
        else:
            return [0, 0, 0, 0, 0, 0, 0]

    # Set up environment
    env = BoxLiftingVelDSSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        tasks_left=None,  # use defaults
        tasks_right=None,  # use defaults
        ref_frame=ref_frame,
        collisionConfig={"file": "collisionModel.xml"},
        fixedInitState=True,
        taskCombinationMethod="sum",
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


if __name__ == "__main__":
    # Choose setup
    setup_type = "bl-ika-vel"  # idle, bl-ika-pos, bl-ika-vel, bl-ds-pos, bl-ds-vel
    common_hparam = dict(
        physicsEngine="Bullet",  # Bullet or Vortex
        graphFileName="gBoxLifting_trqCtrl.xml",  # gBoxLifting_trqCtrl or gBoxLifting_posCtrl
        dt=1 / 100.0,
        max_steps=int(16 * 100),
        ref_frame="basket",  # world, table, or slider
        checkJointLimits=False,
    )

    if setup_type == "idle":
        env, policy = create_idle_setup(**common_hparam)
    elif setup_type == BoxLiftingPosIKActivationSim.name:
        env, policy = create_pos_ika_setup(**common_hparam)
    elif setup_type == BoxLiftingVelIKActivationSim.name:
        env, policy = create_vel_ika_setup(**common_hparam)
    elif setup_type == BoxLiftingPosDSSim.name:
        env, policy = create_pos_mps_setup(**common_hparam)
    elif setup_type == BoxLiftingVelDSSim.name:
        env, policy = create_vel_mps_setup(**common_hparam)
    else:
        raise pyrado.ValueErr(
            given=setup_type,
            eq_constraint=f"idle, {BoxLiftingPosIKActivationSim.name}, {BoxLiftingVelIKActivationSim.name}, "
            f"{BoxLiftingPosDSSim.name}, {BoxLiftingVelDSSim.name}",
        )

    # Simulate and plot
    print("observations:\n", env.obs_space.labels)
    done, param, state = False, None, None
    while not done:
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=False, video=True),
            stop_on_done=False,
            eval=True,
            max_steps=common_hparam["max_steps"],
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
