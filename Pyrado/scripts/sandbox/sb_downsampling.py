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
Test the action downsampling wrapper.
"""
import numpy as np

from matplotlib import pyplot as plt
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.sampling.rollout import rollout
from pyrado.policies.special.environment_specific import QBallBalancerPDCtrl, QQubeSwingUpAndBalanceCtrl
from pyrado.utils.data_types import RenderMode


def create_qbb_setup(factor, dt, max_steps):
    # Set up environment
    init_state = np.array([0, 0, 0.1, 0.1, 0, 0, 0, 0])
    env = QBallBalancerSim(dt=dt, max_steps=max_steps)
    env = ActNormWrapper(env)

    # Set up policy
    policy = QBallBalancerPDCtrl(env.spec)

    # Simulate
    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
        render_mode=RenderMode(video=True),
        max_steps=max_steps,
    )
    act_500Hz = ro.actions

    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt * factor), init_state=init_state),
        render_mode=RenderMode(video=True),
        max_steps=int(max_steps / factor),
    )
    act_100Hz = ro.actions

    env = DownsamplingWrapper(env, factor)
    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
        render_mode=render_mode,
        max_steps=max_steps,
    )
    act_500Hz_w = ro.actions

    # Time in seconds
    time_500Hz = np.linspace(0, int(len(act_500Hz) * dt), int(len(act_500Hz)))
    time_100Hz = np.linspace(0, int(len(act_100Hz) * dt), int(len(act_100Hz)))
    time_500Hz_w = np.linspace(0, int(len(act_500Hz_w) * dt), int(len(act_500Hz_w)))

    # Plot
    _, axs = plt.subplots(nrows=2)
    for i in range(2):
        axs[i].plot(time_500Hz, act_500Hz[:, i], label="500 Hz (original)")
        axs[i].plot(time_100Hz, act_100Hz[:, i], label="100 Hz", ls="--")
        axs[i].plot(time_500Hz_w, act_500Hz_w[:, i], label="500 Hz (wrapped)", ls="--")
        axs[i].legend()
        axs[i].set_ylabel(env.act_space.labels[i])
    axs[1].set_xlabel("time [s]")


def create_qq_setup(factor, dt, max_steps, render_mode):
    # Set up environment
    init_state = np.array([0.1, 0.0, 0.0, 0.0])
    env = QQubeSwingUpSim(dt=dt, max_steps=max_steps)
    env = ActNormWrapper(env)

    # Set up policy
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # Simulate
    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
        render_mode=render_mode,
        max_steps=max_steps,
    )
    act_500Hz = ro.actions

    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt * factor), init_state=init_state),
        render_mode=render_mode,
        max_steps=int(max_steps / factor),
    )
    act_100Hz = ro.actions

    env = DownsamplingWrapper(env, factor)
    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
        render_mode=render_mode,
        max_steps=max_steps,
    )
    act_500Hz_w = ro.actions

    # Time in seconds
    time_500Hz = np.linspace(0, int(len(act_500Hz) * dt), int(len(act_500Hz)))
    time_100Hz = np.linspace(0, int(len(act_100Hz) * dt), int(len(act_100Hz)))
    time_500Hz_w = np.linspace(0, int(len(act_500Hz_w) * dt), int(len(act_500Hz_w)))

    # Plot
    _, ax = plt.subplots(nrows=1)
    ax.plot(time_500Hz, act_500Hz, label="500 Hz (original)")
    ax.plot(time_100Hz, act_100Hz, label="100 Hz", ls="--")
    ax.plot(time_500Hz_w, act_500Hz_w, label="500 Hz (wrapped)", ls="--")
    ax.legend()
    ax.set_ylabel(env.act_space.labels)
    ax.set_xlabel("time [s]")


if __name__ == "__main__":
    common_hparam = dict(
        factor=5,  # don't change this
        dt=1 / 500.0,  # don't change this
        max_steps=1500,  # don't change this
        render_mode=RenderMode(video=False),
    )
    create_qbb_setup(**common_hparam)
    # create_qq_setup(**common_hparam)
    plt.show()


def create_qq_setup(factor, dt, max_steps):
    # Set up environment
    init_state = np.array([0.1, 0.0, 0.0, 0.0])
    env = QQubeSwingUpSim(dt=dt, max_steps=max_steps)
    env = ActNormWrapper(env)

    # Set up policy
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # Simulate
    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
        render_mode=RenderMode(video=True),
        max_steps=max_steps,
    )
    act_500Hz = ro.actions

    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt * factor), init_state=init_state),
        render_mode=RenderMode(video=True),
        max_steps=int(max_steps / factor),
    )
    act_100Hz = ro.actions
    act_100Hz_zoh = np.repeat(act_100Hz, 5, axis=0)

    env = DownsamplingWrapper(env, factor)
    ro = rollout(
        env,
        policy,
        reset_kwargs=dict(domain_param=dict(dt=dt), init_state=init_state),
        render_mode=RenderMode(video=True),
        max_steps=max_steps,
    )
    act_500Hz_wrapped = ro.actions

    # Plot
    _, ax = plt.subplots(nrows=1)
    ax.plot(act_500Hz, label="500 Hz (original)")
    ax.plot(act_100Hz_zoh, label="100 Hz (zoh)")
    ax.plot(act_500Hz_wrapped, label="500 Hz (wrapped)")
    ax.legend()
    ax.set_ylabel(env.act_space.labels)
    ax.set_xlabel("time steps")
    plt.show()


if __name__ == "__main__":
    common_hparam = dict(
        factor=5,  # don't change this
        dt=1 / 500.0,  # don't change this
        max_steps=1000,  # don't change this
    )
    create_qbb_setup(**common_hparam)
    create_qq_setup(**common_hparam)
