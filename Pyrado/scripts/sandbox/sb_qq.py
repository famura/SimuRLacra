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
Test predefined energy-based swing-up controller on the Quanser Qube with observation noise.
"""

import pyrado
from pyrado.environments.mujoco.quanser_qube import QQubeStabMujocoSim, QQubeSwingUpMujocoSim
from pyrado.environments.pysim.quanser_qube import QQubeStabSim, QQubeSwingUpSim
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    dt = 1 / 500.0
    max_steps = 10_000_000
    if args.env_name == "qq-su":
        env = QQubeSwingUpSim(dt=dt, max_steps=max_steps)
    elif args.env_name == "qq-su-mujoco":
        env = QQubeSwingUpMujocoSim(dt=dt, max_steps=max_steps)
    elif args.env_name == "qq-st":
        env = QQubeStabSim(dt=dt, max_steps=max_steps)
    elif args.env_name == "qq-st-mujoco":
        env = QQubeStabMujocoSim(dt=dt, max_steps=max_steps)
    else:
        raise pyrado.ValueErr(
            given_name="--env_name",
            given=args.env_name,
            eq_constraint="'qq-su', 'qq-su-mujoco', 'qq-st', or 'qq-st-mujoco'",
        )
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # Simulate
    done, param, state = False, None, None
    while not done:
        ro = rollout(
            env,
            policy,
            render_mode=RenderMode(text=False, video=args.render, render=args.render),
            eval=True,
            reset_kwargs=dict(domain_param=param, init_state=state),
        )
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, state, param = after_rollout_query(env, policy, ro)
