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
Replay a pre-recorded rollout using the simulations' renderer
"""
import time

import pyrado
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.spaces import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_rollouts_from_dir


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if args.dir is None:
        raise pyrado.ValueErr(msg="Please provide a directory using -d or --dir")

    # Load the rollouts and select one
    rollouts, file_names = load_rollouts_from_dir(args.dir)
    if len(rollouts) == 1:
        rollout = rollouts[0]
    else:
        if not isinstance(args.iter, int):
            raise pyrado.TypeErr(given=args, expected_type=int)
        rollout = rollouts[args.iter]

    # TODO in the future: select via rollout.rollout_info["env_name"]
    if args.env_name.lower() == "wam-jsc":  # avoid loading mujoco
        from pyrado.environments.mujoco.wam_jsc import WAMJointSpaceCtrlSim

        env = WAMJointSpaceCtrlSim(num_dof=7)
        env.init_space = BoxSpace(-pyrado.inf, pyrado.inf, shape=env.init_space.shape)

    elif args.env_name.lower() == QQubeSwingUpSim.name:
        env = QQubeSwingUpSim(dt=1 / 50)

    else:
        raise pyrado.ValueErr(given=args.env_name, eq_constraint=f"wam-jsc, or {QQubeSwingUpSim.name}")

    done = False
    while not done:
        # Simulate like in rollout()
        for step in rollout:
            # Display step by step like rollout()
            t_start = time.time()
            env.state = step.state

            do_sleep = True
            if pyrado.mujoco_loaded:
                from pyrado.environments.mujoco.base import MujocoSimEnv

                # Use reset() to hammer the current state into MuJoCo at evey step
                env.reset(step.state)

                if isinstance(env, MujocoSimEnv):
                    # MuJoCo environments seem to crash on time.sleep()
                    do_sleep = False

            env.render(RenderMode(video=True))

            if do_sleep:
                # Measure time spent and sleep if needed
                t_end = time.time()
                t_sleep = env.dt + t_start - t_end
                if t_sleep > 0:
                    time.sleep(t_sleep)

            # Stop when recorded rollout stopped
            if step.done:
                break

        if input("Stop replaying? [y / any other] ").lower() == "y":
            break
