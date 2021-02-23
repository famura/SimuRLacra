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

import pytest
import numpy as np

from pyrado.environment_wrappers.observation_velfilter import ObsVelFiltWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout
from pyrado.spaces.singular import SingularStateSpace
from pyrado.utils.math import rmse


@pytest.mark.wrapper
@pytest.mark.parametrize("plot", [False, pytest.param(True, marks=pytest.mark.visualization)])
def test_velocity_filter(plot: bool):
    # Set up environment
    env_gt = QQubeSwingUpSim(dt=1 / 500.0, max_steps=350)
    env_gt.init_space = SingularStateSpace(np.array([0.1, np.pi / 2, 3.0, 0]))
    env_filt = ObsVelFiltWrapper(env_gt, idcs_pos=["theta", "alpha"], idcs_vel=["theta_dot", "alpha_dot"])

    # Set up policy
    policy = IdlePolicy(env_gt.spec)

    # Simulate
    ro_gt = rollout(env_gt, policy)
    ro_filt = rollout(env_filt, policy)

    # Filter the observations of the last rollout
    theta_dot_gt = ro_gt.observations[:, 4]
    alpha_dot_gt = ro_gt.observations[:, 5]
    theta_dot_filt = ro_filt.observations[:, 4]
    alpha_dot_filt = ro_filt.observations[:, 5]

    assert theta_dot_filt[0] != pytest.approx(theta_dot_gt[0])  # can't be equal since we set an init vel of 3 rad/s
    assert alpha_dot_filt[0] == pytest.approx(alpha_dot_gt[0], abs=1e-4)

    # Compute the error
    rmse_theta = rmse(theta_dot_gt, theta_dot_filt)
    rmse_alpha = rmse(alpha_dot_gt, alpha_dot_filt)

    if plot:
        from matplotlib import pyplot as plt

        # Plot the filtered signals versus the orignal observations
        plt.rc("text", usetex=True)
        fix, axs = plt.subplots(2, figsize=(16, 9))
        axs[0].plot(theta_dot_gt, label=r"$\dot{\theta}_{true}$")
        axs[0].plot(theta_dot_filt, label=r"$\dot{\theta}_{filt}$")
        axs[1].plot(alpha_dot_gt, label=r"$\dot{\alpha}_{true}$")
        axs[1].plot(alpha_dot_filt, label=r"$\dot{\alpha}_{filt}$")

        axs[0].set_title(rf"RMSE($\theta$): {rmse_theta}")
        axs[0].set_ylabel(r"$\dot{\theta}$ [rad/s]")
        axs[0].legend()
        axs[1].set_title(rf"RMSE($\alpha$): {rmse_alpha}")
        axs[1].set_xlabel("time steps")
        axs[1].set_ylabel(r"$\dot{\alpha}$ [rad/s]")
        axs[1].legend()
        plt.show()
