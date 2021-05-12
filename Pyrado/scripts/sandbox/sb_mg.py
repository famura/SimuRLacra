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
from pyrado.environments.rcspysim.mini_golf import MiniGolfIKSim
from pyrado.policies.feed_forward.dummy import IdlePolicy
from pyrado.policies.feed_forward.poly_time import PolySplineTimePolicy
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(4)


def create_idle_setup(physicsEngine, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = MiniGolfIKSim(
        usePhysicsNode=False,
        physicsEngine=physicsEngine,
        dt=dt,
        max_steps=max_steps,
        ref_frame=ref_frame,
        checkJointLimits=checkJointLimits,
        fixedInitState=True,
    )

    # Set up policy
    policy = IdlePolicy(env.spec)  # don't move at all

    return env, policy


def create_ik_setup(physicsEngine, dt, max_steps, ref_frame, checkJointLimits):
    # Set up environment
    env = MiniGolfIKSim(
        usePhysicsNode=True,
        physicsEngine=physicsEngine,
        dt=dt,
        max_steps=max_steps,
        ref_frame=ref_frame,
        checkJointLimits=checkJointLimits,
        fixedInitState=True,
    )

    # Set up policy
    policy = PolySplineTimePolicy(
        env.spec, dt, t_end=2.0, cond_lvl="vel", cond_final=[0.6, 0.0], overtime_behavior="zero"
    )

    return env, policy


if __name__ == "__main__":
    # Choose setup
    setup_type = "ik"  # idle, or ika
    physicsEngine = "Bullet"  # Bullet or Vortex
    dt = 1 / 100.0
    max_steps = int(15 / dt)
    ref_frame = "world"  # world
    checkJointLimits = False
    randomize = False

    if setup_type == "idle":
        env, policy = create_idle_setup(physicsEngine, dt, max_steps, ref_frame, checkJointLimits)
    elif setup_type == "ik":
        env, policy = create_ik_setup(physicsEngine, dt, max_steps, ref_frame, checkJointLimits)
    else:
        raise pyrado.ValueErr(given=setup_type, eq_constraint="'idle' or 'ik'")

    if randomize:
        dp_nom = env.get_nominal_domain_param()
        randomizer = DomainRandomizer(
            UniformDomainParam(name="ball_mass", mean=dp_nom["ball_mass"], halfspan=dp_nom["ball_mass"] / 5),
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
