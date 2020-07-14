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
Test Linear Policy with RBF Features for the WAM Ball-in-a-cup task.
"""
import torch as to
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

import pyrado
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.mujoco.wam import WAMBallInCupSim
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.environment_specific import DualRBFLinearPolicy
from pyrado.utils.data_types import RenderMode
from pyrado.policies.features import RBFFeat
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


def compute_trajectory(weights, time, width):
    centers = np.linspace(0, 1, weights.shape[0]).reshape(1, -1)  # RBF center locations
    diffs = time - centers

    # Features
    w = np.exp(- diffs**2/(2*width))
    wd = - (diffs/width)*w

    w_sum = np.sum(w, axis=1, keepdims=True)
    wd_sum = np.sum(wd, axis=1, keepdims=True)

    # Normalized features
    pos_features = w/w_sum
    vel_features = (wd*w_sum - w*wd_sum)/w_sum**2

    # Trajectory
    q = pos_features@weights
    qd = vel_features@weights

    # Check gradient computation with finite difference approximation
    for i in range(q.shape[1]):
        qd_approx = np.gradient(q[:, i], 1/len(time))
        assert np.allclose(qd_approx, qd[:, i], rtol=1e-3, atol=1e-3)

    return q, qd


def compute_trajectory_pyrado(weights, time, width):
    weights = to.from_numpy(weights)
    time = to.tensor(time, requires_grad=True)
    rbf = RBFFeat(num_feat_per_dim=weights.shape[0],
                  bounds=(np.array([0.]), np.array([1.])),
                  scale=1/(2*width))
    pos_feat = rbf(time)
    q = pos_feat@weights

    # Explicit
    vel_feat_E = rbf.derivative(time)
    qd_E = vel_feat_E@weights

    # Autograd
    q1, q2, q3 = q.t()
    q1.backward(to.ones((1750,)), retain_graph=True)
    q1d = time.grad.clone()
    time.grad.fill_(0.)
    q2.backward(to.ones((1750,)), retain_graph=True)
    q2d = time.grad.clone()
    time.grad.fill_(0.)
    q3.backward(to.ones((1750,)))
    q3d = time.grad.clone()
    qd = to.cat([q1d, q2d, q3d], dim=1)

    # Check similarity
    assert to.norm(qd_E - qd) < 1e-6

    return q, qd


def check_feat_equality():
    weights = np.random.normal(0, 1, (5, 3))
    time = np.linspace(0, 1, 1750).reshape(-1, 1)
    width = 0.0035
    q1, qd1 = compute_trajectory_pyrado(weights, time, width)
    q2, qd2 = compute_trajectory(weights, time, width)

    assert q1.size() == q2.shape
    assert qd1.size() == qd2.shape

    is_q_equal = np.allclose(q1.detach().numpy(), q2)
    is_qd_equal = np.allclose(qd1.detach().numpy(), qd2)

    correct = is_q_equal and is_qd_equal

    if not correct:
        _, axs = plt.subplots(2)
        axs[0].set_title('positions - solid: pyrado, dashed: reference')
        axs[0].plot(q1.detach().numpy())
        axs[0].set_prop_cycle(None)
        axs[0].plot(q2, ls='--')
        axs[1].set_title('velocities - solid: pyrado, dashed: reference, dotted: finite difference')
        axs[1].plot(qd1.detach().numpy())
        axs[1].set_prop_cycle(None)
        axs[1].plot(qd2, ls='--')
        if is_q_equal:  # q1 and a2 are the same
            finite_diff = np.diff(np.concatenate([np.zeros((1, 3)), q2], axis=0)*500., axis=0)  # init with 0, 500Hz
            axs[1].plot(finite_diff, c='k', ls=':')
        plt.show()

    return correct


def eval_damping():
    """ Plot joint trajectories for different joint damping parameters """
    # Load experiment and remove possible randomization wrappers
    ex_dir = ask_for_experiment()
    env, policy, _ = load_experiment(ex_dir)
    env = inner_env(env)
    env.domain_param = WAMBallInCupSim.get_nominal_domain_param()

    data = []
    t = []
    dampings = [0., 1e-2, 1e-1, 1e0]
    print_cbt(f'Run policy for damping coefficients: {dampings}')
    for d in dampings:
        env.reset(domain_param=dict(joint_damping=d))
        ro = rollout(env, policy, render_mode=RenderMode(video=False), eval=True)
        t.append(ro.env_infos['t'])
        data.append(ro.env_infos['qpos'])

    fig, ax = plt.subplots(3, sharex='all')
    ls = ['k-', 'b--', 'g-.', 'r:']  # line style setting for better visibility
    for i, idx in enumerate([1, 3, 5]):
        for j in range(len(dampings)):
            ax[i].plot(t[j], data[j][:, idx], ls[j], label=f'damping: {dampings[j]}')
            if i == 0:
                ax[i].legend()
        ax[i].set_ylabel(f'joint {idx} pos [rad]')
    ax[2].set_xlabel('time [s]')
    plt.suptitle('Evaluation of joint damping coefficient')
    plt.show()


def rollout_dummy_rbf_policy():
    # Environment
    env = WAMBallInCupSim(max_steps=1750, task_args=dict(sparse_rew_fcn=True))

    # Stabilize around initial position
    env.reset(domain_param=dict(cup_scale=1., rope_length=0.3103, ball_mass=0.021))
    act = np.zeros((6,))  # desired deltas from the initial pose
    for i in range(500):
        env.step(act)
        env.render(mode=RenderMode(video=True))

    # Apply DualRBFLinearPolicy
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.]), np.array([1.])))
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam, dim_mask=1)
    done, param = False, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(video=True), eval=True, reset_kwargs=dict(domain_param=param))
        print_cbt(f'Return: {ro.undiscounted_return()}', 'g', bright=True)
        done, _, param = after_rollout_query(env, policy, ro)

    # Retrieve infos from rollout
    t = ro.env_infos['t']
    des_pos_traj = ro.env_infos['qpos_des']  # (max_steps,7) ndarray
    pos_traj = ro.env_infos['qpos']
    des_vel_traj = ro.env_infos['qvel_des']  # (max_steps,7) ndarray
    vel_traj = ro.env_infos['qvel']
    ball_pos = ro.env_infos['ball_pos']
    cup_pos = ro.env_infos['cup_pos']

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(3, sharex='all')
    for i, idx in enumerate([1, 3, 5]):
        ax[i].plot(t, des_pos_traj[:, idx], label=f'qpos_des {idx}')
        ax[i].plot(t, pos_traj[:, idx], label=f'qpos {idx}')
        ax[i].legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs=ball_pos[:, 0], ys=ball_pos[:, 1], zs=ball_pos[:, 2], color='blue', label='Ball')
    ax.scatter(xs=ball_pos[-1, 0], ys=ball_pos[-1, 1], zs=ball_pos[-1, 2], color='blue', label='Ball final')
    ax.plot(xs=cup_pos[:, 0], ys=cup_pos[:, 1], zs=cup_pos[:, 2], color='red', label='Cup')
    ax.scatter(xs=cup_pos[-1, 0], ys=cup_pos[-1, 1], zs=cup_pos[-1, 2], color='red', label='Cup final')
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(elev=16., azim=-7.)
    plt.show()


if __name__ == '__main__':
    # Fix seed for reproducibility
    pyrado.set_seed(101)

    # Check for function equality
    if check_feat_equality():
        print_cbt('The two methods to compute the trajectory yield equal results.', 'g')
    else:
        print_cbt('The two methods to compute the trajectory do not yield equal results.', 'r')

    # Plot damping coefficient comparison
    # eval_damping()

    # Apply DualRBFLinearPolicy and plot the joint states over the desired ones
    rollout_dummy_rbf_policy()
