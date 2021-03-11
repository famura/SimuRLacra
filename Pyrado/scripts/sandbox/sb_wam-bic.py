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
Test Linear Policy with RBF Features for the WAM ball-in-the-cup task.
"""
import torch as to
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.

import pyrado
from pyrado.environments.mujoco.wam_bic import WAMBallInCupSim
from pyrado.policies.special.dual_rfb import DualRBFLinearPolicy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from pyrado.policies.features import RBFFeat
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.input_output import print_cbt


def compute_trajectory(weights, time, width):
    centers = np.linspace(0, 1, weights.shape[0]).reshape(1, -1)  # RBF center locations
    diffs = time - centers

    # Features
    w = np.exp(-(diffs ** 2) / (2 * width))
    wd = -(diffs / width) * w

    w_sum = np.sum(w, axis=1, keepdims=True)
    wd_sum = np.sum(wd, axis=1, keepdims=True)

    # Normalized features
    pos_features = w / w_sum
    vel_features = (wd * w_sum - w * wd_sum) / w_sum ** 2

    # Trajectory
    q = pos_features @ weights
    qd = vel_features @ weights

    # Check gradient computation with finite difference approximation
    for i in range(q.shape[1]):
        qd_approx = np.gradient(q[:, i], 1 / len(time))
        assert np.allclose(qd_approx, qd[:, i], rtol=1e-3, atol=1e-3)

    return q, qd


def compute_trajectory_pyrado(weights, time, width):
    weights = to.from_numpy(weights).to(dtype=to.get_default_dtype())
    time = to.tensor(time, requires_grad=True, dtype=to.get_default_dtype())
    rbf = RBFFeat(num_feat_per_dim=weights.shape[0], bounds=(np.array([0.0]), np.array([1.0])), scale=1 / (2 * width))
    pos_feat = rbf(time)
    q = pos_feat @ weights

    # Explicit
    vel_feat_E = rbf.derivative(time)
    qd_E = vel_feat_E @ weights

    # Autograd
    q_1, q_2, q3 = q.t()
    q_1.backward(to.ones((1750,)), retain_graph=True)
    q_1d = time.grad.clone()
    time.grad.fill_(0.0)
    q_2.backward(to.ones((1750,)), retain_graph=True)
    q_2d = time.grad.clone()
    time.grad.fill_(0.0)
    q3.backward(to.ones((1750,)))
    q3d = time.grad.clone()
    qd = to.cat([q_1d, q_2d, q3d], dim=1)

    # Check similarity
    assert to.norm(qd_E - qd) < 1e-3  # used to be 1e-6 with double precision

    return q, qd


def check_feat_equality():
    weights = np.random.normal(0, 1, (5, 3))
    time = np.linspace(0, 1, 1750).reshape(-1, 1)
    width = 0.0035
    q_1, qd_1 = compute_trajectory_pyrado(weights, time, width)
    q_2, qd_2 = compute_trajectory(weights, time, width)

    assert q_1.size() == q_2.shape
    assert qd_1.size() == qd_2.shape

    is_q_equal = np.allclose(q_1.detach().cpu().numpy(), q_2, atol=1e-6)
    is_qd_equal = np.allclose(qd_1.detach().cpu().numpy(), qd_2, atol=1e-5)

    correct = is_q_equal and is_qd_equal

    if not correct:
        _, axs = plt.subplots(2)
        axs[0].set_title("Joint Positions: pyrado and reference")
        axs[0].plot(q_1.detach().cpu().numpy(), ls="--", label="pyrado")
        axs[0].set_prop_cycle(None)
        axs[0].plot(q_2, ls="-.", label="reference")
        axs[0].legend()
        axs[1].set_title("velocities - solid: pyrado, dashed: reference, dotted: finite difference")
        axs[1].plot(qd_1.detach().cpu().numpy(), ls="--", label="pyrado")
        axs[1].set_prop_cycle(None)
        axs[1].plot(qd_2, ls="-.", label="reference")
        axs[1].legend()

        if is_q_equal:  # q_1 and a2 are the same
            finite_diff = np.diff(np.concatenate([np.zeros((1, 3)), q_2], axis=0) * 500.0, axis=0)  # init with 0, 500Hz
            axs[1].plot(finite_diff, c="k", ls=":")

        plt.show()
    return correct


def eval_damping():
    """ Plot joint trajectories for different joint damping parameters """
    # Environment
    env = WAMBallInCupSim(num_dof=7, max_steps=1500)

    # Policy (random init)
    policy_hparam = dict(num_feat_per_dim=12, bounds=(np.array([0.0]), np.array([1.0])))
    policy = DualRBFLinearPolicy(env.spec, policy_hparam, dim_mask=2)

    # Do the rolllouts
    t_all = []
    qpos_all = []
    dp_vals = [0.0, 0.01, 0.1, 0.5, 1.0]
    print_cbt(f"Run policy for damping coefficients: {dp_vals}")
    for dpv in dp_vals:
        env.reset(
            domain_param=dict(
                joint_1_damping=dpv,
                joint_2_damping=dpv,
                joint_3_damping=dpv,
                joint_4_damping=dpv,
                joint_5_damping=dpv,
                joint_6_damping=dpv,
                joint_7_damping=dpv,
            )
        )
        ro = rollout(env, policy, render_mode=RenderMode(video=False), eval=True)
        t_all.append(ro.time[:-1])
        qpos_all.append(ro.env_infos["qpos"])

    # Plot
    fig, ax = plt.subplots(nrows=env.num_dof, sharex="all", figsize=(16, 7))
    for i, idx_joint in enumerate([dof for dof in range(env.num_dof)]):
        ax[i].set_prop_cycle(color=plt.get_cmap("cividis")(np.linspace(0, 1, env.num_dof)))
        ax[i].set_ylabel(f"joint {idx_joint+1} pos [rad]")
        for j in range(len(dp_vals)):
            ax[i].plot(t_all[j], qpos_all[j][:, idx_joint], ls="--", label=f"d = {dp_vals[j]}")
            if i == 0:
                ax[i].legend(ncol=len(dp_vals))
    ax[-1].set_xlabel("time [s]")
    plt.suptitle("Evaluation of joint damping coefficients")
    plt.show()


def eval_dryfriction():
    """ Plot joint trajectories for different joint stiction parameters """
    # Environment
    env = WAMBallInCupSim(num_dof=7, max_steps=1500)

    # Policy (random init)
    policy_hparam = dict(num_feat_per_dim=12, bounds=(np.array([0.0]), np.array([1.0])))
    policy = DualRBFLinearPolicy(env.spec, policy_hparam, dim_mask=2)

    # Do the rolllouts
    t_all = []
    qpos_all = []
    dp_vals = [0.0, 0.3, 0.6, 0.9, 1.2]
    print_cbt(f"Run policy for stiction coefficients: {dp_vals}")
    for dpv in dp_vals:
        env.reset(
            domain_param=dict(
                joint_1_dryfriction=dpv,
                joint_2_dryfriction=dpv,
                joint_3_dryfriction=dpv,
                joint_4_dryfriction=dpv,
                joint_5_dryfriction=dpv,
                joint_6_dryfriction=dpv,
                joint_7_dryfriction=dpv,
            )
        )
        ro = rollout(env, policy, render_mode=RenderMode(video=False), eval=True)
        t_all.append(ro.time[:-1])
        qpos_all.append(ro.env_infos["qpos"])

    # Plot
    fig, ax = plt.subplots(nrows=env.num_dof, sharex="all", figsize=(16, 7))
    for i, idx_joint in enumerate([dof for dof in range(env.num_dof)]):
        ax[i].set_prop_cycle(color=plt.get_cmap("cividis")(np.linspace(0, 1, env.num_dof)))
        ax[i].set_ylabel(f"joint {idx_joint+1} pos [rad]")
        for j in range(len(dp_vals)):
            ax[i].plot(t_all[j], qpos_all[j][:, idx_joint], ls="--", label=f"s = {dp_vals[j]}")
            if i == 0:
                ax[i].legend(ncol=len(dp_vals))
    ax[-1].set_xlabel("time [s]")
    plt.suptitle("Evaluation of joint stiction coefficients")
    plt.show()


def rollout_dummy_rbf_policy_7dof():
    # Environment
    env = WAMBallInCupSim(num_dof=7, max_steps=1750, task_args=dict(sparse_rew_fcn=True))

    # Stabilize around initial position
    env.reset(domain_param=dict(cup_scale=1.0, rope_length=0.3103, ball_mass=0.021))
    act = np.zeros((6,))  # desired deltas from the initial pose
    for i in range(500):
        env.step(act)
        env.render(mode=RenderMode(video=True))

    # Apply DualRBFLinearPolicy
    policy_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.0]), np.array([1.0])))
    policy = DualRBFLinearPolicy(env.spec, policy_hparam, dim_mask=1)
    done, param = False, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(video=True), eval=True, reset_kwargs=dict(domain_param=param))
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, _, param = after_rollout_query(env, policy, ro)

    # Retrieve infos from rollout
    t = ro.time
    des_pos_traj = ro.env_infos["qpos_des"]
    pos_traj = ro.env_infos["qpos"]
    des_vel_traj = ro.env_infos["qvel_des"]
    vel_traj = ro.env_infos["qvel"]
    ball_pos = ro.env_infos["ball_pos"]
    cup_pos = ro.env_infos["cup_pos"]

    # Plot trajectories of the directly controlled joints and their corresponding desired trajectories
    fig, ax = plt.subplots(3, sharex="all")
    for i, idx in enumerate([1, 3, 5]):
        ax[i].plot(t, des_pos_traj[:, idx], label=f"qpos_des {idx}")
        ax[i].plot(t, pos_traj[:, idx], label=f"qpos {idx}")
        ax[i].legend()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs=ball_pos[:, 0], ys=ball_pos[:, 1], zs=ball_pos[:, 2], color="blue", label="Ball")
    ax.scatter(xs=ball_pos[-1, 0], ys=ball_pos[-1, 1], zs=ball_pos[-1, 2], color="blue", label="Ball final")
    ax.plot(xs=cup_pos[:, 0], ys=cup_pos[:, 1], zs=cup_pos[:, 2], color="red", label="Cup")
    ax.scatter(xs=cup_pos[-1, 0], ys=cup_pos[-1, 1], zs=cup_pos[-1, 2], color="red", label="Cup final")
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=16.0, azim=-7.0)
    plt.show()


def rollout_dummy_rbf_policy_4dof():
    # Environment
    env = WAMBallInCupSim(
        num_dof=4,
        max_steps=3000,
        # Note, when tuning the task args: the `R` matrices are now 4x4 for the 4 dof WAM
        task_args=dict(R=np.zeros((4, 4)), R_dev=np.diag([0.2, 0.2, 1e-2, 1e-2])),
    )

    # Stabilize ball and print out the stable state
    env.reset()
    act = np.zeros(env.spec.act_space.flat_dim)
    for i in range(1500):
        env.step(act)
        env.render(mode=RenderMode(video=True))

    # Printing out actual positions for 4-dof (..just needed to setup the hard-coded values in the class)
    print("Ball pos:", env.sim.data.get_body_xpos("ball"))
    print("Cup goal:", env.sim.data.get_site_xpos("cup_goal"))
    print("Cup bottom:", env.sim.data.get_site_xpos("cup_bottom"))
    print("Joint pos (incl. first rope angle):", env.sim.data.qpos[:5])

    # Apply DualRBFLinearPolicy and plot the joint states over the desired ones
    rbf_hparam = dict(num_feat_per_dim=7, bounds=(np.array([0.0]), np.array([1.0])))
    policy = DualRBFLinearPolicy(env.spec, rbf_hparam, dim_mask=2)
    done, param = False, None
    while not done:
        ro = rollout(env, policy, render_mode=RenderMode(video=True), eval=True, reset_kwargs=dict(domain_param=param))
        print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
        done, _, param = after_rollout_query(env, policy, ro)


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Set the seed
    pyrado.set_seed(0)

    # Check for function equality
    if check_feat_equality():
        print_cbt("The two methods to compute the trajectory yield equal results.", "g")
    else:
        print_cbt("The two methods to compute the trajectory do not yield equal results.", "r")

    if args.mode.lower() == "damping":
        eval_damping()

    elif args.mode.lower() == "stiction":
        eval_dryfriction()

    elif args.mode.lower() == "7dof":
        rollout_dummy_rbf_policy_7dof()

    elif args.mode.lower() == "4dof":
        rollout_dummy_rbf_policy_4dof()

    else:
        raise pyrado.ValueErr(given=args.mode, eq_constraint="damping, stiction, 7dof, or 4dof")
