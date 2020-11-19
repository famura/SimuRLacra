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
Test predefined energy-based swing-up controller on the Quanser Qube with observation noise.
"""
from scipy.ndimage import gaussian_filter1d
from matplotlib import pyplot as plt

from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


if __name__ == '__main__':
    plt.rc('text', usetex=True)

    # Set up environment
    env = QQubeSwingUpSim(dt=1/500., max_steps=3500)
    env = GaussianObsNoiseWrapper(env, noise_std=[0., 0., 0., 0., 2., 0])  # only noise on theta_dot [rad/s]

    # Set up policy
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # Simulate
    ro = rollout(env, policy, render_mode=RenderMode(text=False, video=False), eval=True)

    # Filter the observations of the last rollout
    theta_dot = ro.observations[:, 4]
    alpha_dot = ro.observations[:, 5]
    theta_dot_filt_3 = gaussian_filter1d(theta_dot, 3)
    theta_dot_filt_5 = gaussian_filter1d(theta_dot, 5)
    alpha_dot_filt_3 = gaussian_filter1d(alpha_dot, 3)
    alpha_dot_filt_5 = gaussian_filter1d(alpha_dot, 5)

    # Plot the filtered signals versus the orignal observations
    fix, axs = plt.subplots(2, figsize=(16, 8))
    axs[0].plot(theta_dot, label=r'$\dot{\theta}$')
    axs[0].plot(theta_dot_filt_3, label=r'$\dot{\theta}_{filt}, \sigma=3$')
    axs[0].plot(theta_dot_filt_5, label=r'$\dot{\theta}_{filt}, \sigma=5$')
    axs[1].plot(alpha_dot, label=r'$\dot{\alpha}$')
    axs[1].plot(alpha_dot_filt_3, label=r'$\dot{\alpha}_{filt}, \sigma=3$')
    axs[1].plot(alpha_dot_filt_5, label=r'$\dot{\alpha}_{filt}, \sigma=5$')

    axs[0].set_title(r'Gaussian 1D filter on noisy $\theta$ signal')
    axs[1].set_ylabel(r'$\dot{\theta}$ [rad/s]')
    axs[0].legend()
    axs[1].set_title(r'Gaussian 1D filter on clean $\alpha$ signal')
    axs[1].set_xlabel('time steps')
    axs[1].set_ylabel(r'$\dot{\alpha}$ [rad/s]')
    axs[1].legend()
    plt.show()
