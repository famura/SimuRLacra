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
Script to test the different ways of continuously blending movement primitives
"""
import math
import numpy as np
from matplotlib import pyplot as plt

import rcsenv
from pyrado.environments.rcspysim.mp_blending import MPBlendingSim
from pyrado.policies.special.time import TimePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


rcsenv.setLogLevel(0)


# Define a policy
def policy_fcn(t: float):
    # return [1., 0., 0., 0.]
    # return [1., 1., 0., 0.]
    # return [1., 1., 1., 0.]
    # return [0.5, 0.5, 0.5, 0.5]
    # return [0.3, 0.1, 0.8, 0.5]
    # return [0.3, (math.cos(0.3*math.pi*t) + 1)/10., abs(math.cos(0.5*math.pi*t)), 0.1]
    # return [0.3, (math.cos(0.3*math.pi*t) + 1)/10., abs(math.cos(0.5*math.pi*t)), int(t > 5)]
    return [0.3*(1e-6 + 1 - int(8 < t < 9)),
            (math.cos(0.3*math.pi*t) + 1)/10.*(1e-6 + 1 - int(8 < t < 9)),
            abs(math.cos(0.5*math.pi*t))*(1e-6 + 1 - int(8 < t < 9)),
            int(t > 5)*(1e-6 + 1 - int(8 < t < 9))]


def sum_variant(action_model_type, mps, dt, max_steps, physics_engine, render_mode):
    # Set up environment
    env = MPBlendingSim(
        action_model_type=action_model_type,
        mps=mps,
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        collisionAvoidanceIK=False,
        taskCombinationMethod='sum'
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=render_mode, stop_on_done=False)


def mean_variant(action_model_type, mps, dt, max_steps, physics_engine, render_mode):
    # Set up environment
    env = MPBlendingSim(
        action_model_type=action_model_type,
        mps=mps,
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        collisionAvoidanceIK=False,
        taskCombinationMethod='mean'
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=render_mode, stop_on_done=False)


def softmax_variant(action_model_type, mps, dt, max_steps, physics_engine, render_mode):
    # Set up environment
    env = MPBlendingSim(
        action_model_type=action_model_type,
        mps=mps,
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        collisionAvoidanceIK=False,
        taskCombinationMethod='softmax'
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=render_mode, stop_on_done=False)


def product_variant(action_model_type, mps, dt, max_steps, physics_engine, render_mode):
    # Set up environment
    env = MPBlendingSim(
        action_model_type=action_model_type,
        mps=mps,
        physicsEngine=physics_engine,
        dt=dt,
        max_steps=max_steps,
        collisionAvoidanceIK=False,
        taskCombinationMethod='product'
    )

    # Set up policy
    policy = TimePolicy(env.spec, policy_fcn, dt)

    # Simulate
    return rollout(env, policy, render_mode=render_mode, stop_on_done=False)


def create_lin_pos_mps():
    return [
        {'function': 'lin', 'errorDynamics': 50*np.eye(2), 'goal': np.array([-1., -1.])},
        {'function': 'lin', 'errorDynamics': 50*np.eye(2), 'goal': np.array([-1., 1.])},
        {'function': 'lin', 'errorDynamics': 50*np.eye(2), 'goal': np.array([1., -1.])},
        {'function': 'lin', 'errorDynamics': 50*np.eye(2), 'goal': np.array([1., 1.])},
    ]


def create_nlin_pos_mps():
    return [
        {
            'function': 'msd_nlin',
            'attractorStiffness': 50., 'mass': 2., 'damping': 50.,
            'goal': np.array([-1., -1.]),  # position of the lower left goal
        },
        {
            'function': 'msd_nlin',
            'attractorStiffness': 50., 'mass': 2., 'damping': 50.,
            'goal': np.array([-1., 1.]),  # position of the upper left goal
        },
        {
            'function': 'msd_nlin',
            'attractorStiffness': 50., 'mass': 2., 'damping': 50.,
            'goal': np.array([1., -1.]),  # position of the lower right goal
        },
        {
            'function': 'msd_nlin',
            'attractorStiffness': 50., 'mass': 2., 'damping': 50.,
            'goal': np.array([1., 1.]),  # position of the upper right goal
        }
    ]


def _plot_annotated_goals(ax):
    pos_x = [-1, -1, 1, 1]
    pos_y = [-1, 1, -1, 1]
    ax.scatter(pos_x, pos_y, s=6)
    for i, z in enumerate(zip(pos_x, pos_y)):
        ax.annotate(f'{i}', z, ha='center', va='bottom', xytext=(4, 4), textcoords='offset points')  # in points


if __name__ == '__main__':
    # Define movement primitives
    # mps = create_msd_nlin_pos_mps()
    mps = create_lin_pos_mps()

    # Define hyper-parameters
    common_hparams = dict(
        action_model_type='ik_activation',  # ds_activation or ik_activation
        mps=mps, dt=0.005, max_steps=2000, physics_engine='Vortex', render_mode=RenderMode(video=False)
    )

    # Run
    rollouts = dict(
        sum=sum_variant(**common_hparams),
        mean=mean_variant(**common_hparams),
        softmax=softmax_variant(**common_hparams),
        product=product_variant(**common_hparams)
    )

    # Plot
    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    dim_act = 4  # rollouts['sum'].actions.shape[1]
    gs = fig.add_gridspec(nrows=dim_act + 2, ncols=2)

    ax_trajs_xy = fig.add_subplot(gs[:-2, 0])
    ax_trajs_xy.title.set_text('Trajectories')
    ax_trajs_xy.set_xlabel('x')
    ax_trajs_xy.set_ylabel('y')
    for i, (name, ro) in enumerate(rollouts.items()):
        ax_trajs_xy.quiver(ro.observations[:-1, 0], ro.observations[:-1, 1],
                           ro.observations[1:, 0] - ro.observations[:-1, 0],
                           ro.observations[1:, 1] - ro.observations[:-1, 1],
                           scale_units='xy', angles='xy', scale=1, headwidth=4, label=name, color=f'C{i%10}')
        plt.legend(loc='upper center')
    _plot_annotated_goals(ax_trajs_xy)
    ax_trajs_xy.grid()
    ax_trajs_xy.axis('equal')
    ax_trajs_xy.set_xlim(-1.05, 1.05)
    ax_trajs_xy.set_ylim(-1.05, 1.05)

    ax_trajs_x = fig.add_subplot(gs[-2, :])
    ax_trajs_x.title.set_text('Trajectories')
    ax_trajs_x.set_ylabel('x')
    plt.legend(loc='upper right')
    for i, (name, ro) in enumerate(rollouts.items()):
        ax_trajs_x.plot(ro.observations[:, 0], label=name, color=f'C{i%10}')
    ax_trajs_y = fig.add_subplot(gs[-1, :])
    ax_trajs_y.set_ylabel('y')
    ax_trajs_y.set_xlabel('time step')
    plt.legend(loc='upper right')
    for i, (name, ro) in enumerate(rollouts.items()):
        ax_trajs_y.plot(ro.observations[:, 1], label=name, color=f'C{i%10}')

    action_labels = ['$a_0$ to goal 0', '$a_1$ to goal 1', '$a_2$ goal 2', '$a_3$ to goal 3']
    for idx_act in range(dim_act):
        ax_act = fig.add_subplot(gs[idx_act, 1])

        for _, ro in rollouts.items():
            ax_act.plot(ro.actions[:, idx_act])
            # Only plot one the trials since they all got the same actions
            break

        if idx_act == 0:
            ax_act.title.set_text('Actions')
        if idx_act == dim_act - 1:
            ax_act.set_xlabel('time step')
        ax_act.set_ylabel(action_labels[idx_act])

    plt.show()
