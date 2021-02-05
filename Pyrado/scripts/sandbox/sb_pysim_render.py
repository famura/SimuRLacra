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
Test predefined energy-based controller to make the Quanser Cart-Pole swing up or balancing task.
"""
import numpy as np

import pyrado
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    dt = args.dt if args.dt is not None else 0.01

    if args.env_name == QCartPoleSwingUpSim.name:
        env = QCartPoleSwingUpSim(dt=dt, max_steps=int(10 / dt), wild_init=False)
        state = np.array([0, 87 / 180 * np.pi, 0, 0])

    elif args.env_name == QQubeSwingUpSim.name:
        env = QQubeSwingUpSim(dt=dt, max_steps=int(10 / dt))
        state = np.array([5 / 180 * np.pi, 87 / 180 * np.pi, 0, 0])

    elif args.env_name == QBallBalancerSim.name:
        env = QBallBalancerSim(dt=dt, max_steps=int(10 / dt))
        state = np.array([2 / 180 * np.pi, 2 / 180 * np.pi, 0.1, -0.08, 0, 0, 0, 0])

    elif args.env_name == OneMassOscillatorSim.name:
        env = OneMassOscillatorSim(dt=dt, max_steps=int(10 / dt))
        state = np.array([-0.7, 0])

    elif args.env_name == PendulumSim.name:
        env = PendulumSim(dt=dt, max_steps=int(10 / dt))
        state = np.array([87 / 180 * np.pi, 0])

    else:
        raise pyrado.ValueErr(
            given=args.env_name,
            eq_constraint=f"{QCartPoleSwingUpSim.name}, {QQubeSwingUpSim.name}, {QBallBalancerSim.name}",
        )

    policy = IdlePolicy(env.spec)

    # Simulate
    done, param = False, None
    while not done:
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=True, video=True),
            eval=True,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        # print_domain_params(env.domain_param)
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
