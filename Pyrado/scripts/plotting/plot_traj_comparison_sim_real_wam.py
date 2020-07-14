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

    # Load real trajectories
    qpos_real = np.load(osp.join(ex_dir, 'qpos_real.npy'))
    qvel_real = np.load(osp.join(ex_dir, 'qvel_real.npy'))

    # Load the policy and the environment
    env, policy, _ = load_experiment(ex_dir, args)

    # Get nominal environment
    env = remove_all_dr_wrappers(env)
    env.domain_param = env.get_nominal_domain_param()

    # Fix seed for reproducibility
    pyrado.set_seed(args.seed)

    # Use the recorded initial state from the real system
    init_state = env.init_space.sample_uniform()
    init_state[:7] = qpos_real[0, :]

    # Do rollout in simulation
    ro = rollout(env, policy, eval=True, render_mode=RenderMode(video=False), reset_kwargs=dict(init_state=init_state))
    t, qpos_sim, qvel_sim = ro.env_infos['t'], ro.env_infos['qpos'], ro.env_infos['qvel']
    err_deg = 180/np.pi*(qpos_real - qpos_sim)
    rmse = np.sqrt(np.mean(np.power(err_deg, 2), axis=0)).round(2)

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex='all', constrained_layout=True)
    fig.canvas.set_window_title('Trajectory Comparison')
    fig.suptitle(f'RMSE: $q_1$ = {rmse[1]} deg, $q_3$ = {rmse[3]} deg, $q_5$ {rmse[5]} deg')

    for i, idx in enumerate([1, 3, 5]):
        ax[i, 0].plot(t, 180/np.pi*qpos_sim[:, idx], label='sim')
        ax[i, 0].plot(t, 180/np.pi*qpos_real[:, idx], label='real')
        ax[i, 1].plot(t, 180/np.pi*qvel_sim[:, idx], label='sim')
        ax[i, 1].plot(t, 180/np.pi*qvel_real[:, idx], label='real')
        ax[i, 0].set_ylabel(rf'$q_{idx} [deg]$')
        ax[i, 1].set_ylabel(rf'$\dot{{{{q}}}}_{idx} [deg/s]$')
        if i == 0:
            ax[i, 0].legend()
            ax[i, 1].legend()

    ax[2, 0].set_xlabel('$t$ [s]')
    ax[2, 1].set_xlabel('$t$ [s]')

    plt.show()
