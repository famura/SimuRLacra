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
Plot trajectories of the directly controlled joints of the Barrett WAM.
The plot always includes the trajectories from the MuJoCo simulation as baseline and optionally
    1) a trajectory recorded on the real Barrett WAM and
    2) the desired trajectories specified by the policy.
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

    # Get the experiment's directory to load from if not given as command line argument
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load real trajectories
    mode = ''
    real_data_exists = True
    try:
        while mode not in ['ep', 'sb']:
            mode = input('Pass ep for episodic and sb for step-based control mode: ').lower()
        qpos_real = np.load(osp.join(ex_dir, f'qpos_real_{mode}.npy'))
        qvel_real = np.load(osp.join(ex_dir, f'qvel_real_{mode}.npy'))
    except FileNotFoundError:
        real_data_exists = False
        print_cbt(f'Did not find a recorded real trajectory (qpos_real_{mode} and qvel_real_{mode}) for this policy. '
                  f'Run deployment/run_policy_wam.py to get real-world trajectories.',
                  'y', bright=True)

    # Load the policy and the environment
    env, policy, _ = load_experiment(ex_dir, args)

    # Get nominal environment
    env = remove_all_dr_wrappers(env)
    env.domain_param = env.get_nominal_domain_param()
    env.stop_on_collision = False

    # Fix seed for reproducibility
    pyrado.set_seed(args.seed)

    # Use the recorded initial state from the real system
    init_state = env.init_space.sample_uniform()
    if real_data_exists:
        if input('Use the recorded initial state from the real system? [y] / n ').lower() == '' or 'y':
            init_state[:env.num_dof] = qpos_real[0, :]

    # Define indices of actuated joints
    act_idcs = [1, 3, 5] if env.num_dof == 7 else [1, 3]

    # Do rollout in simulation
    ro = rollout(env, policy, eval=True, render_mode=RenderMode(video=False), reset_kwargs=dict(init_state=init_state))
    t = ro.env_infos['t']
    qpos_sim, qvel_sim = ro.env_infos['qpos'], ro.env_infos['qvel']
    qpos_des, qvel_des = ro.env_infos['qpos_des'], ro.env_infos['qvel_des']

    plot_des = False
    if input('Plot the desired joint states and velocities (i.e. the policy features)? [y] / n ') == '' or 'y':
        plot_des = True

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(nrows=len(act_idcs), ncols=2, figsize=(12, 8), sharex='all', constrained_layout=True)
    fig.canvas.set_window_title('Trajectory Comparison')

    # Compute the RMSE (root mean squared error) in degree
    if real_data_exists:
        err_deg = 180/np.pi*(qpos_real - qpos_sim)
        rmse = np.sqrt(np.mean(np.power(err_deg, 2), axis=0)).round(2)
        suptitle = f'RMSE: q_1 = {rmse[1]} deg, q_3 = {rmse[3]} deg'
        if env.num_dof == 7:
            suptitle = suptitle + f', q_5 {rmse[5]} deg'
        fig.suptitle(suptitle)

    for i, idx in enumerate(act_idcs):
        ax[i, 0].plot(t, 180/np.pi*qpos_sim[:, idx], label='sim')
        ax[i, 1].plot(t, 180/np.pi*qvel_sim[:, idx], label='sim')

        if real_data_exists:
            ax[i, 0].plot(t, 180/np.pi*qpos_real[:, idx], label='real')
            ax[i, 1].plot(t, 180/np.pi*qvel_real[:, idx], label='real')

        if plot_des:
            ax[i, 0].plot(t, 180/np.pi*qpos_des[:, idx], label='des', ls='--')
            ax[i, 1].plot(t, 180/np.pi*qvel_des[:, idx], label='des', ls='--')

        ax[i, 0].set_ylabel(f'q_{idx} [deg]')
        ax[i, 1].set_ylabel(f'qd_{idx} [deg/s]')

    ax[0, 0].legend()
    ax[0, 1].legend()

    ax[0, 0].set_title('joint position')
    ax[0, 1].set_title('joint velocity')

    ax[len(act_idcs) - 1, 0].set_xlabel('t [s]')
    ax[len(act_idcs) - 1, 1].set_xlabel('t [s]')

    plt.show()
