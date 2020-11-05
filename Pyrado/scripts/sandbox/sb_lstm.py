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
This is a very basic script to test the functionality of Rcs & RcsPySim & Pyrado using a robotic ball-on-plate setup
using an untrained recurrent policy.
"""
import torch as to

import rcsenv
from matplotlib import pyplot as plt
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate5DSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.domain_randomization.utils import print_domain_params
from pyrado.policies.recurrent.rnn import LSTMPolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


rcsenv.setLogLevel(4)

if __name__ == '__main__':
    # Set up environment
    dt = 0.01
    env = BallOnPlate5DSim(
        physicsEngine='Vortex',  # Bullet or Vortex
        dt=dt,
        max_steps=2000,
    )
    env = ActNormWrapper(env)
    print_domain_params(env.domain_param)

    # Set up policy
    policy = LSTMPolicy(env.spec, 20, 1)
    policy.init_param()

    # Simulate
    ro = rollout(env, policy, render_mode=RenderMode(video=True), stop_on_done=True)

    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), sharex='all', tight_layout=True)
    axs[0].plot(ro.observations[:, 1], label='plate y pos')
    axs[1].plot(ro.observations[:, 2], label='plate z pos')
    axs[0].legend()
    axs[1].legend()
    plt.show()

    ro.torch()

    # Simulate in the policy's eval mode
    ev_act, _ = policy(ro.observations[:-1, ...], ro.hidden_states)
    print(f'Difference in actions between recordings in simulation and the policy evaluation mode:\n'
          f'{to.max(to.abs(ev_act - ro.actions))}')
