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
import math

import rcsenv
import torch as to

import pyrado
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.rcspysim.mini_golf import MiniGolfIKSim
from pyrado.policies.features import FeatureStack, const_feat
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.policies.feed_forward.dummy import IdlePolicy
from pyrado.policies.feed_forward.poly_time import PolySplineTimePolicy
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(2)


def create_idle_setup(physicsEngine: str, dt: float, max_steps: int, checkJointLimits: bool):
    # Set up environment
    env = MiniGolfIKSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        dt=dt,
        max_steps=max_steps,
        checkJointLimits=checkJointLimits,
        fixedInitState=True,
        observeForceTorque=True,
    )

    # Set up policy
    policy = IdlePolicy(env.spec)  # don't move at all

    return env, policy


def create_pst_setup(physicsEngine: str, dt: float, max_steps: int, checkJointLimits: bool):
    # Set up environment
    relativeZdTask = True
    print_cbt(f"relativeZdTask = {relativeZdTask}", "c", bright=True)
    env = MiniGolfIKSim(
        relativeZdTask=relativeZdTask,
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        dt=dt,
        max_steps=max_steps,
        checkJointLimits=checkJointLimits,
        fixedInitState=True,
        observeForceTorque=False,
        collisionAvoidanceIK=True,
    )

    # Set up policy
    if relativeZdTask:
        policy_hparam = dict(
            t_end=1.0,
            cond_lvl="vel",
            # Zd (rel), Y (rel), Zdist (abs), PHI (abs), THETA (abs)
            cond_final=[
                [0.0, 0.0, 0.01, math.pi / 2, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            cond_init=[
                [-7.0, 0.0, 0.01, math.pi / 2, 0.0],
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

    return env, policy


def create_lin_setup(physicsEngine: str, dt: float, max_steps: int, checkJointLimits: bool):
    # Set up environment
    env = MiniGolfIKSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        dt=dt,
        max_steps=max_steps,
        checkJointLimits=checkJointLimits,
        fixedInitState=True,
    )

    # Set up policy
    policy = LinearPolicy(env.spec, FeatureStack([const_feat]))
    policy.param_values = to.tensor([0.6, 0.0, 0.03])  # X (abs), Y (rel), Z (abs), C (abs)

    return env, policy


def create_time_setup(physicsEngine: str, dt: float, max_steps: int, checkJointLimits: bool):
    # Set up environment
    env = MiniGolfJntPosSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        dt=dt,
        max_steps=max_steps,
        checkJointLimits=checkJointLimits,
        fixedInitState=True,
    )

    # Set up policy
    def policy_fcn(t: float):
        return [
            10 / 180 * math.pi,
            10 / 180 * math.pi,  # same as init config
            10 / 180 * math.pi + 45.0 / 180.0 * math.pi * math.sin(2.0 * math.pi * 0.2 * t),
        ]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    return env, policy


if __name__ == "__main__":
    # Choose setup
    setup_type = "pst"  # idle, pst, or lin
    physicsEngine = "Bullet"  # Bullet or Vortex
    dt = 1 / 100.0
    max_steps = int(13 / dt)
    checkJointLimits = True
    randomize = True

    if setup_type == "idle":
        env, policy = create_idle_setup(physicsEngine, dt, max_steps, checkJointLimits)
    elif setup_type == "pst":
        env, policy = create_pst_setup(physicsEngine, dt, max_steps, checkJointLimits)
    elif setup_type == "lin":
        env, policy = create_lin_setup(physicsEngine, dt, max_steps, checkJointLimits)
    elif setup_type == "time":
        env, policy = create_time_setup(physicsEngine, dt, max_steps, checkJointLimits)
    else:
        raise pyrado.ValueErr(given=setup_type, eq_constraint="idle, pst, lin, or time")

    if randomize:
        dp_nom = env.get_nominal_domain_param()
        randomizer = DomainRandomizer(
            UniformDomainParam(name="ball_radius", mean=dp_nom["ball_radius"], halfspan=dp_nom["ball_radius"] / 5),
            UniformDomainParam(name="ball_mass", mean=dp_nom["ball_mass"], halfspan=dp_nom["ball_mass"] / 5),
            UniformDomainParam(name="club_mass", mean=dp_nom["club_mass"], halfspan=dp_nom["club_mass"] / 5),
            UniformDomainParam(
                name="ball_friction_coefficient",
                mean=dp_nom["ball_friction_coefficient"],
                halfspan=dp_nom["ball_friction_coefficient"] / 4,
                clip_lo=0,
            ),
            UniformDomainParam(
                name="ball_rolling_friction_coefficient",
                mean=dp_nom["ball_rolling_friction_coefficient"],
                halfspan=dp_nom["ball_rolling_friction_coefficient"] / 3,
                clip_lo=0,
            ),
            UniformDomainParam(
                name="ground_friction_coefficient",
                mean=dp_nom["ground_friction_coefficient"],
                halfspan=dp_nom["ground_friction_coefficient"] / 4,
                clip_lo=0,
            ),
            UniformDomainParam(name="ball_slip", mean=dp_nom["ball_slip"], halfspan=dp_nom["ball_slip"] / 2, clip_lo=0),
            UniformDomainParam(
                name="ground_slip", mean=dp_nom["ground_slip"], halfspan=dp_nom["ground_slip"] / 2, clip_lo=0
            ),
            UniformDomainParam(name="obstacleleft_pos_offset_x", mean=0, halfspan=0.01),
            UniformDomainParam(name="obstacleleft_pos_offset_y", mean=-0.1, halfspan=0.01),
            UniformDomainParam(name="obstacleleft_rot_offset_c", mean=15 / 180 * math.pi, halfspan=1 / 180 * math.pi),
            UniformDomainParam(name="obstacleright_pos_offset_x", mean=0, halfspan=0.01),
            UniformDomainParam(name="obstacleright_pos_offset_y", mean=0.1, halfspan=0.01),
            UniformDomainParam(name="obstacleright_rot_offset_c", mean=-15 / 180 * math.pi, halfspan=1 / 180 * math.pi),
        )
        env = DomainRandWrapperLive(env, randomizer)

    # Simulate and plot
    print(env.obs_space)
    done, param, state = False, None, None
    while not done:
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=False, video=True),
            eval=True,
            max_steps=max_steps,
            reset_kwargs=dict(domain_param=param, init_state=state),
            stop_on_done=False,
        )
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
