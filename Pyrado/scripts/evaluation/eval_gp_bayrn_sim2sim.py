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
Script for the paper plot the GP's posterior after a Bayesian Domain Randomization sim-to-sim experiment
"""

import joblib
import os.path as osp
import torch as to
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

import pyrado
from pyrado.environment_wrappers.domain_randomization import MetaDomainRandWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.gaussian_process import render_singletask_gp
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import ensure_no_subscript


if __name__ == '__main__':
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument('--render3D', action='store_true', default=False, help="render the GP in 3D")
    args = parser.parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    env = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))
    if not typed_env(env, MetaDomainRandWrapper):
        raise pyrado.TypeErr(given_name=env, expected_type=MetaDomainRandWrapper)
    cands = to.load(osp.join(ex_dir, 'candidates.pt'))
    cands_values = to.load(osp.join(ex_dir, 'candidates_values.pt')).unsqueeze(1)
    bounds = to.load(osp.join(ex_dir, 'bounds.pt'))

    dim_cand = cands.shape[1]  # number of domain distribution parameters
    if dim_cand%2 != 0:
        raise pyrado.ShapeErr(msg='The dimension of domain distribution parameters must be a multiple of 2!')

    # Select dimensions to plot (ignored for 1D mode)
    if len(args.idcs) == 1:
        # Plot 1D
        x_label = ensure_no_subscript(env.mapping[args.idcs[0]][0])  # could override manually here
        y_label = r'$\hat{J}^{\textrm{real}}$'
        fig, ax = plt.subplots(1, figsize=(12, 8), constrained_layout=True)

    elif len(args.idcs) == 2:
        x_label = ensure_no_subscript(env.mapping[args.idcs[0]][0])  # could override manually here
        y_label = ensure_no_subscript(env.mapping[args.idcs[1]][0])  # could override manually here

        if not args.render3D:
            # Plot 2D
            hm_fig_size = (pyrado.figsize_IEEE_1col_square[0]*0.75, pyrado.figsize_IEEE_1col_square[0]*0.75)
            cb_fig_size = (pyrado.figsize_IEEE_1col_square[0]*0.15, pyrado.figsize_IEEE_1col_square[0]*0.55)

            fig_hm_mean, ax_hm_mean = plt.subplots(1, figsize=hm_fig_size, constrained_layout=True)
            fig_cb_mean, ax_cb_mean = plt.subplots(1, figsize=cb_fig_size, constrained_layout=True)
            fig_hm_std, ax_hm_std = plt.subplots(1, figsize=hm_fig_size, constrained_layout=True)
            fig_cb_std, ax_cb_std = plt.subplots(1, figsize=cb_fig_size, constrained_layout=True)
            ax = [ax_hm_mean, ax_cb_mean, ax_hm_std, ax_cb_std]

        else:
            # Plot 3D
            fig = plt.figure(figsize=(12, 8))
            ax = Axes3D(fig)

    else:
        raise pyrado.ValueErr(msg='Select exactly 1 or 2 indices!')

    # Nice color map from seaborn
    # hm_cmap = sns.cubehelix_palette(light=.9, dark=.1, reverse=True, as_cmap=True)
    # hm_cmap = sns.light_palette("muted_navy", reverse=True, as_cmap=True)
    hm_cmap = ListedColormap(sns.color_palette("YlGnBu", n_colors=100)[::-1])
    # hm_cmap = ListedColormap(sns.color_palette("YlOrRd", n_colors=100)[::-1])
    # hm_cmap = ListedColormap(sns.color_palette("OrRd", n_colors=100)[::-1])
    # scat_cmap = LinearSegmentedColormap.from_list('white_to_gray', [(1., 1., 1.), (.4, .4, .4)], N=256)
    scat_cmap = LinearSegmentedColormap.from_list('light_to_dark_gray', [(.8, .8, .8), (.2, .2, .2)], N=256)

    render_singletask_gp(
        ax, cands, cands_values, min_gp_obsnoise=1e-5,
        # data_x_min=bounds[0, args.idcs], data_x_max=bounds[1, args.idcs],
        idcs_sel=args.idcs, x_label=x_label, y_label=y_label, z_label=r'$\hat{J}^{\textrm{real}}$',
        heatmap_cmap=hm_cmap, num_stds=2, resolution=151, legend_data_cmap=scat_cmap, show_legend_data=args.verbose,
        show_legend_posterior=True, show_legend_std=True, render3D=args.render3D,
    )

    if len(args.idcs) == 2 and not args.render3D:
        # Plot the ground truth domain parameter configuration
        ax_hm_mean.scatter(0.026, 0.097, c='firebrick', marker='o', s=60)  # forestgreen
        ax_hm_std.scatter(0.026, 0.097, c='firebrick', marker='o', s=60)  # forestgreen

    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            if len(args.idcs) == 1 or args.render3D:
                fig.savefig(osp.join(ex_dir, f'gp-posterior-ret-mean.{fmt}'), dpi=500)
            if len(args.idcs) == 2 and not args.render3D:
                fig_hm_mean.savefig(osp.join(ex_dir, f'gp-posterior-ret-mean-hm.{fmt}'), dpi=500)
                fig_cb_mean.savefig(osp.join(ex_dir, f'gp-posterior-ret-mean-cb.{fmt}'), dpi=500)
                fig_hm_std.savefig(osp.join(ex_dir, f'gp-posterior-ret-std-hm.{fmt}'), dpi=500)
                fig_cb_std.savefig(osp.join(ex_dir, f'gp-posterior-ret-std-cb.{fmt}'), dpi=500)

    plt.show()
