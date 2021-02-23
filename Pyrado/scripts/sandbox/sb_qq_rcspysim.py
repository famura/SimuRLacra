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
This is a very basic script to test the functionality of Rcs & RcsPySim & Pyrado using the Quanser Qube setup.
"""
import numpy as np

from pyrado.environments.rcspysim.quanser_qube import QQubeRcsSim
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.plotting.rollout_based import plot_observations_actions_rewards
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


if __name__ == "__main__":
    # Set up environment
    dt = 1 / 5000.0
    max_steps = 5000
    env = QQubeRcsSim(physicsEngine="Bullet", dt=dt, max_steps=max_steps, max_dist_force=None)  # Bullet or Vortex
    print_domain_params(env.domain_param)

    # Set up policy
    policy = TimePolicy(env.spec, lambda t: [1.0], dt)  # constant acceleration with 1. rad/s**2

    # Simulate
    ro = rollout(
        env,
        policy,
        render_mode=RenderMode(video=True),
        reset_kwargs=dict(init_state=np.array([0, 3 / 180 * np.pi, 0, 0.0])),
    )

    # Plot
    print(f"After {max_steps*dt} s of accelerating with 1. rad/s**2, we should be at {max_steps*dt} rad/s")
    print(f"Difference: {max_steps*dt - ro.observations[-1][2]} rad/s (mind the swinging pendulum)")
    plot_observations_actions_rewards(ro)
