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
Run a policy (trained in simulation) on real Barret WAM.
"""
import os.path as osp
import numpy as np

import pyrado
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.barrett_wam.wam import WAMBallInCupRealEpisodic, WAMBallInCupRealStepBased
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.experiments import wrap_like_other_env, load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the policy (trained in simulation) and the environment (for constructing the real-world counterpart)
    env_sim, policy, _ = load_experiment(ex_dir, args)

    # Detect the correct real-world counterpart and create it
    if env_sim.name == "wam-bic":  # use hard-coded name to avoid loading mujoco_py by loading WAMBallInCupSim
        # If `max_steps` (or `dt`) are not explicitly set using `args`, use the same as in the simulation
        max_steps = args.max_steps if args.max_steps < pyrado.inf else env_sim.max_steps
        dt = args.dt if args.dt is not None else env_sim.dt
        mode = ""
        while mode not in ["ep", "sb"]:
            mode = input("Pass ep for episodic and sb for step-based control mode: ").lower()
        if mode == "sb":
            env_real = WAMBallInCupRealStepBased(
                observe_ball=env_sim.observe_ball,
                observe_cup=env_sim.observe_cup,
                dt=dt,
                max_steps=max_steps,
                num_dof=inner_env(env_sim).num_dof,
            )
        else:
            env_real = WAMBallInCupRealEpisodic(dt=dt, max_steps=max_steps, num_dof=inner_env(env_sim).num_dof)
    else:
        raise pyrado.ValueErr(given=env_sim.name, eq_constraint="wam-bic")

    # Wrap the real environment in the same way as done during training
    env_real = wrap_like_other_env(env_real, env_sim)

    # Run on device
    done = False
    print_cbt("Running loaded policy ...", "c", bright=True)
    while not done:
        ro = rollout(env_real, policy, eval=True)
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        np.save(osp.join(ex_dir, f"qpos_real_{mode}.npy"), env_real.qpos_real)
        np.save(osp.join(ex_dir, f"qvel_real_{mode}.npy"), env_real.qvel_real)
        print_cbt(f"Saved trajectory into qpos_real_{mode}.npy and qvel_real_{mode}.npy", "g")
        done, _, _ = after_rollout_query(env_real, policy, ro)
