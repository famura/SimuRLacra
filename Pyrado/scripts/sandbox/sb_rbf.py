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

import numpy as np
import torch as to
from matplotlib import pyplot as plt
from matplotlib import ticker

import pyrado
from pyrado.policies.features import FeatureStack, RBFFeat
from pyrado.policies.linear import LinearPolicy
from pyrado.spaces import BoxSpace
from pyrado.utils.data_types import EnvSpec


if __name__ == '__main__':
    # Define some arbitrary EnvSpec
    obs_space = BoxSpace(bound_lo=np.array([-5., -12.]), bound_up=np.array([10., 6.]))
    act_space = BoxSpace(bound_lo=np.array([-1.]), bound_up=np.array([1.]))
    spec = EnvSpec(obs_space, act_space)

    num_fpd = 5
    num_eval_points = 500

    policy_hparam = dict(
        feats=FeatureStack([RBFFeat(num_feat_per_dim=num_fpd, bounds=obs_space.bounds, scale=None)])
    )
    policy = LinearPolicy(spec, **policy_hparam)

    eval_grid_0 = to.linspace(-5., 10, num_eval_points)
    eval_grid_1 = to.linspace(-12., 6, num_eval_points)
    eval_grid = to.stack([eval_grid_0, eval_grid_1], dim=1)

    feat_vals = to.empty(num_eval_points, num_fpd*obs_space.flat_dim)
    # Feed evaluation samples one by one
    for i, obs in enumerate(eval_grid):
        feat_vals[i, :] = policy.eval_feats(obs)

    feat_vals_batch = policy.eval_feats(eval_grid)

    if (feat_vals == feat_vals_batch).all():
        feat_vals = feat_vals_batch
    else:
        raise pyrado.ValErr(msg='Batch mode failed')

    if num_eval_points <= 50:
        # Plot the feature values over the feature index
        _, axs = plt.subplots(obs_space.flat_dim, 1, figsize=(10, 8), tight_layout=False)
        for i, fv in enumerate(feat_vals):
            axs[0].plot(np.arange(num_fpd), fv[:num_fpd].numpy(), label=i)
            axs[1].plot(np.arange(num_fpd), fv[num_fpd:].numpy(), label=i)
        axs[0].legend(title='eval point', ncol=num_eval_points//2,
                      loc='upper center', bbox_to_anchor=(0., 1.1, 1., 0.3), mode='expand')
        axs[0].set_title('reconstructed input dim 1')
        axs[1].set_title('reconstructed input dim 2')
        axs[1].set_xlabel('feature')
        axs[0].set_ylabel('activation')
        axs[1].set_ylabel('activation')
        axs[0].xaxis.set_major_locator(ticker.MaxNLocator(num_fpd, integer=True))
        axs[1].xaxis.set_major_locator(ticker.MaxNLocator(num_fpd, integer=True))

    # Plot the feature values over the input samples
    _, axs = plt.subplots(obs_space.flat_dim, 1, figsize=(10, 8), tight_layout=False)
    for i in range(obs_space.flat_dim):
        for j, fv in enumerate(feat_vals[:, i*num_fpd:(i + 1)*num_fpd].T):
            axs[i].plot(fv.numpy(), label=j, c=f'C{j%10}')
        axs[i].legend()

    axs[0].set_title('reconstructed input dim 1')
    axs[1].set_title('reconstructed input dim 2')
    axs[1].set_xlabel('input sample')
    axs[0].set_ylabel('activation')
    axs[1].set_ylabel('activation')

    plt.show()
