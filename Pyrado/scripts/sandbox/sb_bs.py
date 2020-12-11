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
Script to test the bi-manual box shelving task using a hard-coded time-based policy
"""
import rcsenv
import pyrado
from pyrado.environments.rcspysim.box_shelving import BoxShelvingPosDSSim, BoxShelvingVelDSSim
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(4)


def create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = BoxShelvingPosDSSim(
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        fixedInitState=False,
        collisionConfig={"file": "collisionModel.xml"},
        checkJointLimits=checkJointLimits,
    )

    # Set up policy
    policy = IdlePolicy(env.spec)  # don't move at all

    return env, policy


def create_position_mps_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits):
    def policy_fcn(t: float):
        return [0, 0, 1, 0, 0, 0.01]  # PG position  # PG orientation  # hand joints

    # Set up environment
    env = BoxShelvingPosDSSim(
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        fixedInitState=True,
        collisionConfig={"file": "collisionModel.xml"},
        checkJointLimits=checkJointLimits,
        collisionAvoidanceIK=True,
        observeVelocities=False,
        observeCollisionCost=True,
        observePredictedCollisionCost=True,
        observeManipulabilityIndex=True,
        observeCurrentManipulability=True,
        observeDynamicalSystemDiscrepancy=True,
        observeTaskSpaceDiscrepancy=True,
        observeForceTorque=True,
        observeDynamicalSystemGoalDistance=True,
    )
    print(env.get_body_position("Box", "", ""))
    print(env.get_body_position("Box", "GoalUpperShelve", ""))
    print(env.get_body_position("Box", "", "GoalUpperShelve"))
    print(env.get_body_position("Box", "GoalUpperShelve", "GoalUpperShelve"))

    # Set up policy
    policy = TimePolicy(env.spec, policy_fcn, dt)

    return env, policy


def create_velocity_mps_setup(
    physicsEngine, graphFileName, dt, max_steps, ref_frame, bidirectional_mps, checkJointLimits
):
    if bidirectional_mps:

        def policy_fcn(t: float):
            if t <= 1:
                return [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            elif t <= 2.0:
                return [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]
            elif t <= 3:
                return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
            elif t <= 4.0:
                return [0.0, 0.0, 0.0, -1.0, 0.0, 0.0]
            elif t <= 5:
                return [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            elif t <= 6.0:
                return [0.0, 0.0, -1.0, 0.0, 0.0, 0.0]
            elif t <= 7:
                return [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
            elif t <= 8:
                return [0.0, -1.0, 0.0, 0.0, 0.0, 0.0]
            elif t <= 9:
                return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            elif t <= 10:
                return [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            else:
                return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    else:

        def policy_fcn(t: float):
            if t <= 1:
                return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
            elif t <= 2.0:
                return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
            elif t <= 3:
                return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
            elif t <= 4.0:
                return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
            elif t <= 5:
                return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
            elif t <= 6.0:
                return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
            elif t <= 7:
                return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
            elif t <= 8:
                return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]
            elif t <= 9:
                return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            elif t <= 10:
                return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            else:
                return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Set up environment
    env = BoxShelvingVelDSSim(
        physicsEngine=physicsEngine,
        graphFileName=graphFileName,
        dt=dt,
        max_steps=max_steps,
        mps_left=None,  # use defaults
        mps_right=None,  # use defaults
        ref_frame=ref_frame,
        bidirectional_mps=bidirectional_mps,
        fixedInitState=True,
        collisionConfig={"file": "collisionModel.xml"},
        checkJointLimits=checkJointLimits,
        observeVelocities=True,
        observeCollisionCost=True,
        observePredictedCollisionCost=True,
        observeManipulabilityIndex=True,
        observeCurrentManipulability=True,
        observeDynamicalSystemDiscrepancy=True,
        observeTaskSpaceDiscrepancy=True,
        observeForceTorque=True,
        observeDynamicalSystemGoalDistance=True,
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy_fcn, dt)

    return env, policy


if __name__ == "__main__":
    # Choose setup
    setup_type = "pos"  # idle, pos, or vel
    bidirectional_mps = False  # only for velocity-level MPs
    physicsEngine = "Bullet"  # Bullet or Vortex
    graphFileName = "gBoxShelving_posCtrl.xml"  # gBoxShelving_trqCtrl or gBoxShelving_posCtrl
    dt = 1 / 100.0
    max_steps = int(12 / dt)
    ref_frame = "upperGoal"  # box or world or upperGoal
    checkJointLimits = True

    if setup_type == "idle":
        env, policy = create_idle_setup(physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits)
    elif setup_type == "pos":
        env, policy = create_position_mps_setup(
            physicsEngine, graphFileName, dt, max_steps, ref_frame, checkJointLimits
        )
    elif setup_type == "vel":
        env, policy = create_velocity_mps_setup(
            physicsEngine, graphFileName, dt, max_steps, ref_frame, bidirectional_mps, checkJointLimits
        )
    else:
        raise pyrado.ValueErr(given=setup_type, eq_constraint="'idle', 'pos', or 'vel'")

    # Simulate and plot
    print("observations:\n", env.obs_space.labels)
    done, param, state = False, None, None
    while not done:
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=False, video=True),
            eval=True,
            max_steps=max_steps,
            stop_on_done=False,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
