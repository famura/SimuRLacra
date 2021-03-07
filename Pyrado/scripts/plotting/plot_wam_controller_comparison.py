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
Compare the trajectories of the directly controlled joints of the Barrett WAM for the two different control approaches:
    1) Episodic
    2) Step-based
"""
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(show_hyper_parameters=args.show_hyperparameters) if args.dir is None else args.dir

    # Load trajectories
    try:
        qpos_episodic = np.load(osp.join(ex_dir, "qpos_real_ep.npy"))
        qpos_stepbased = np.load(osp.join(ex_dir, "qpos_real_sb.npy"))
        qvel_episodic = np.load(osp.join(ex_dir, "qvel_real_ep.npy"))
        qvel_stepbased = np.load(osp.join(ex_dir, "qvel_real_sb.npy"))
    except FileNotFoundError:
        print_cbt(
            "Did not find at least one of the required real trajectories. "
            "You first need to run deployment/run_policy_wam.py with the Episodic and Step-based real WAM "
            "respectively.",
            color="r",
            bright=True,
        )
        raise

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    act_dim = 2 if qpos_episodic.shape[1] == 4 else 3
    act_idcs = [1, 3] if act_dim == 2 else [1, 3, 5]

    fig, ax = plt.subplots(nrows=act_dim, ncols=2, figsize=(12, 8), sharex="all", constrained_layout=True)
    fig.canvas.set_window_title("Trajectory Comparison")

    for i, idx in enumerate(act_idcs):
        # Positions
        ax[i, 0].plot(180 / np.pi * qpos_episodic[:, idx], label="episodic", ls="--")
        ax[i, 0].plot(180 / np.pi * qpos_stepbased[:, idx], label="step-based", ls="--")

        # Velocities
        ax[i, 1].plot(180 / np.pi * qvel_episodic[:, idx], label="episodic", ls="--")
        ax[i, 1].plot(180 / np.pi * qvel_stepbased[:, idx], label="step-based", ls="--")

        ax[i, 0].set_ylabel(f"q_{idx} [deg]")
        ax[i, 1].set_ylabel(f"qd_{idx} [deg/s]", size="large")

    ax[0, 0].legend()
    ax[0, 1].legend()

    ax[0, 0].set_title("joint position")
    ax[0, 1].set_title("joint velocity")

    ax[act_dim - 1, 0].set_xlabel("steps")
    ax[act_dim - 1, 1].set_xlabel("steps")

    plt.show()
