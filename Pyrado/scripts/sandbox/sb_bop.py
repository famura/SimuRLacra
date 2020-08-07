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
Script to test the functionality of Rcs & RcsPySim & Pyrado using a robotic ball-on-plate setup
"""
import math
from matplotlib import pyplot as plt

import rcsenv
import pyrado
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate5DSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


rcsenv.setLogLevel(4)


def create_setup(physics_engine, dt, max_steps, max_dist_force):
    # Set up environment
    env = BallOnPlate5DSim(
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        max_dist_force=max_dist_force
    )
    env = ActNormWrapper(env)
    print_domain_params(env.domain_param)

    # Set up policy
    def policy_fcn(t: float):
        return [
            0.0,  # x_ddot_plate
            0.5*math.sin(2.*math.pi*5*t),  # y_ddot_plate
            5.*math.cos(2.*math.pi/5.*t),  # z_ddot_plate
            0.0,  # alpha_ddot_plate
            0.0,  # beta_ddot_plate
        ]

    policy = TimePolicy(env.spec, policy_fcn, dt)

    return env, policy


if __name__ == '__main__':
    # Initialize
    fig, axs = plt.subplots(3, figsize=(8, 12), sharex='col', tight_layout=True)

    # Try to run several possible cases
    for pe in ['Bullet', 'Vortex']:
        print_cbt(f'Running with {pe} physics engine', 'c', bright=True)

        if rcsenv.supportsPhysicsEngine(pe):
            env, policy = create_setup(pe, dt=0.01, max_steps=1000, max_dist_force=0.)

            # Simulate
            ro = rollout(env, policy, render_mode=RenderMode(video=True), seed=0)

            # Render plots
            axs[0].plot(ro.observations[:, 0], label=pe)
            axs[1].plot(ro.observations[:, 1], label=pe)
            axs[2].plot(ro.observations[:, 2], label=pe)
            axs[0].legend()
            axs[1].legend()
            axs[2].legend()

    # Show plots
    axs[0].set_title('gBotKuka.xml')
    axs[0].set_ylabel('plate x pos')
    axs[1].set_ylabel('plate y pos')
    axs[2].set_ylabel('plate z pos')
    axs[2].set_xlabel('time steps')
    plt.show()
