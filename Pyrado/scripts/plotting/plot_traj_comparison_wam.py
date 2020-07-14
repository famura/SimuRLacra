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
Plot a trajectory recorded on the real Barrett WAM and compare it to a simulation, starting from the same initial pose.
"""
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt

import pyrado
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load trajectories
    qpos = np.load(osp.join(ex_dir, 'qpos.npy'))  # sim
    qpos_des = np.load(osp.join(ex_dir, 'qpos_des.npy'))  # sim
    qvel = np.load(osp.join(ex_dir, 'qvel.npy'))  # sim
    qvel_des = np.load(osp.join(ex_dir, 'qvel_des.npy'))  # sim
    try:
        data_exists = True
        qpos_real_ff = np.load(osp.join(ex_dir, 'qpos_real_ff.npy'))
        qpos_real_pd = np.load(osp.join(ex_dir, 'qpos_real_pd.npy'))
        qvel_real_ff = np.load(osp.join(ex_dir, 'qvel_real_ff.npy'))
        qvel_real_pd = np.load(osp.join(ex_dir, 'qvel_real_pd.npy'))
    except FileNotFoundError:
        data_exists = False
        print_cbt('Did not find a recorded real trajectory (qpos_real and qvel_real) for this policy!',
                  'r', bright=True)

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex='all', constrained_layout=True)
    fig.canvas.set_window_title('Trajectory Comparison')

    for i, idx in enumerate([1, 3, 5]):
        # Positions
        ax[i, 0].plot(180/np.pi*qpos[:, idx], label='qpos', ls='--')
        ax[i, 0].plot(180/np.pi*qpos_des[:, idx], label='qpos_des', ls='--')
        if data_exists:
            ax[i, 0].plot(180/np.pi*qpos_real_ff[:, idx], label='qpos_real_ff')
            ax[i, 0].plot(180/np.pi*qpos_real_pd[:, idx], label='qpos_real_pd')
        # Velocities
        ax[i, 1].plot(180/np.pi*qvel[:, idx], label='qvel', ls='--')
        ax[i, 1].plot(180/np.pi*qvel_des[:, idx], label='qvel_des', ls='--')
        if data_exists:
            ax[i, 1].plot(180/np.pi*qvel_real_ff[:, idx], label='qvel_real_ff')
            ax[i, 1].plot(180/np.pi*qvel_real_pd[:, idx], label='qvel_real_pd')

        ax[i, 0].set_ylabel(rf'$q_{idx} [deg]$')
        ax[i, 1].set_ylabel(rf'$\dot{{{{q}}}}_{idx} [deg/s]$')
        if i == 0:
            ax[i, 0].legend()
            ax[i, 1].legend()

    ax[2, 0].set_xlabel('steps')
    ax[2, 1].set_xlabel('steps')

    plt.show()
