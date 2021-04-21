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
Excite the system using a sin or chirp signal on the actions
"""
from datetime import datetime

import numpy as np
from scipy.signal import chirp

import pyrado
from pyrado.environments.barrett_wam.wam_bic import WAMReal
from pyrado.environments.barrett_wam.wam_jsc import WAMJointSpaceCtrlRealStepBased
from pyrado.environments.mujoco.wam_jsc import WAMJointSpaceCtrlSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.quanser.quanser_cartpole import QCartPoleSwingUpReal
from pyrado.environments.quanser.quanser_qube import QQubeSwingUpReal
from pyrado.policies.special.environment_specific import wam_jsp_7dof_sin
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import after_rollout_query, rollout
from pyrado.spaces import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.functions import skyline
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    dt = args.dt or 1 / 500.0
    t_end = 5.5  # s
    max_steps = int(t_end / dt)  # run for 5s
    check_in_sim = False
    # max_amp = 5.0 / 180 * np.pi  # max. amplitude [rad]
    max_amp = -3.5  # max. amplitude [V]

    # Create the simulated and real environments
    if args.env_name == QQubeSwingUpReal.name:
        env_sim = QQubeSwingUpSim(dt, max_steps)
        env_real = QQubeSwingUpReal(dt, max_steps)
    elif args.env_name == QCartPoleSwingUpReal.name:
        env_sim = QCartPoleSwingUpSim(dt, max_steps)
        env_real = QCartPoleSwingUpReal(dt, max_steps)
    elif args.env_name == WAMReal.name:
        env_sim = WAMJointSpaceCtrlSim(frame_skip=4, num_dof=7, max_steps=max_steps)
        env_real = WAMJointSpaceCtrlRealStepBased(num_dof=7, max_steps=max_steps)
    else:
        raise pyrado.ValueErr(
            given=args.env_name, eq_constraint=f"{QQubeSwingUpReal.name}, {QCartPoleSwingUpReal.name} or {WAMReal.name}"
        )
    print_cbt(f"Set the {env_real.name} environment.", "g")

    # Create the policy
    if args.mode.lower() == "chirp":

        def fcn_of_time(t: float):
            act = max_amp * chirp(t, f0=3, f1=0, t1=t_end, method="linear")
            return act.repeat(env_real.act_space.flat_dim)

    elif args.mode.lower() == "skyline":
        t_intvl_space = BoxSpace(dt, 100 * dt, shape=(1,))
        val_space = BoxSpace(-abs(max_amp), abs(max_amp), shape=(1,))
        act_predef = skyline(dt, t_end, t_intvl_space, val_space)[1]

        def fcn_of_time(t: float):
            act = act_predef[int(t / dt)]
            return act.repeat(env_real.act_space.flat_dim)

    elif args.mode.lower() == "sin":

        def fcn_of_time(t: float):
            act = max_amp * np.sin(2 * np.pi * t * 2.0)  # 2 Hz
            return act.repeat(env_real.act_space.flat_dim)

    elif args.mode.lower() == "wam_sin":
        fcn_of_time = wam_jsp_7dof_sin

    else:
        raise pyrado.ValueErr(given=args.mode, eq_constraint="chrip or sin")

    policy = TimePolicy(env_real.spec, fcn_of_time, dt)

    # Simulate before executing
    if check_in_sim:
        print_cbt(f"Running action {args.mode} policy in simulation...", "c")
        done = False
        while not done:
            ro = rollout(env_sim, policy, eval=True, render_mode=RenderMode(video=True))
            done, _, _ = after_rollout_query(env_real, policy, ro)

    # Run on device
    print_cbt(f"Running action {args.mode} policy on the robot...", "c")
    done = False
    while not done:
        ro = rollout(env_real, policy, eval=True)

        if args.save:
            pyrado.save(
                ro,
                "rollout_real.pkl",
                pyrado.TEMP_DIR,
                suffix=datetime.now().strftime(pyrado.timestamp_format),
            )
            print_cbt(f"Saved rollout to {pyrado.TEMP_DIR}", "g")

        done, _, _ = after_rollout_query(env_real, policy, ro)
