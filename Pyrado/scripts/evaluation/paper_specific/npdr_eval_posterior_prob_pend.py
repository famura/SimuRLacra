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
Script to evaluate a posteriors obtained with NPDR and BayesSim in a round-wise fashion with scatter pair-plots showing
samples form the posteriors, as reported in

.. seealso::
    [1] F. Muratore, T. Gruner, F. Wiese, B. Belousov, M. Gienger, J. Peters, "TITLE", VENUE, YEAR
"""
import os

import seaborn as sns
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.plotting.distribution import draw_posterior_heatmap_2d, draw_posterior_scatter_2d
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt
from pyrado.utils.ordering import remove_none_from_list


def _load_experiment(ex_dir: pyrado.PathLike):
    # Load the algorithm
    algo = Algorithm.load_snapshot(ex_dir)
    if not isinstance(algo, (NPDR, BayesSim)):
        raise pyrado.TypeErr(given=algo, expected_type=(NPDR, BayesSim))

    # Load the prior and the data
    prior = pyrado.load("prior.pt", ex_dir)
    data_real = pyrado.load("data_real.pt", ex_dir)

    # Load the posteriors
    posteriors = [SBIBase.load_posterior(ex_dir, idx_round=i, verbose=True) for i in range(algo.num_sbi_rounds)]
    posteriors = remove_none_from_list(posteriors)  # in case the algorithm terminated early

    if data_real.shape[0] > len(posteriors):
        print_cbt(
            f"Found {data_real.shape[0]} data sets but {len(posteriors)} posteriors. Truncated the superfluous data.",
            "y",
        )
        data_real = data_real[: len(posteriors), :]

    # Artificially repeat the data (which was the same for every round) to later be able to use the same code
    data_real = data_real.repeat(len(posteriors), 1)
    assert data_real.shape[0] == len(posteriors)

    return algo, prior, data_real, posteriors


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    args = parser.parse_args()
    plt.rc("text", usetex=args.use_tex)
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_congfigs = [
        dict(
            algo_name=NPDR.name,
            dir=os.path.join(pyrado.EXP_DIR, "pend", "npdr_pb", ""),
            sel_rounds=[0, 2, 7],
        ),
        dict(
            algo_name=BayesSim.name,
            dir=os.path.join(pyrado.EXP_DIR, "pend", "bayessim_pb", ""),
            sel_rounds=[0, 1, 2],
        ),
    ]

    for config in ex_congfigs:
        # Load the algorithm
        algo, prior, data_real, posteriors = _load_experiment(config["dir"])

        # Set the color map
        cmap = plt.get_cmap("turbo")

        # Round-wise processing
        ax_cnt = 0  # for selection
        fig, axs = plt.subplots(
            nrows=1, ncols=3, figsize=pyrado.figsize_CoRL_6perrow_square  # , constrained_layout=True
        )
        for idx, (posterior, data) in enumerate(zip(posteriors, data_real)):
            # Select round or not
            if idx not in config["sel_rounds"]:
                continue

            if args.mode == "scatter":
                # Sample from the posterior
                domain_params, log_probs = SBIBase.eval_posterior(
                    posterior,
                    data.unsqueeze(0),
                    args.num_samples,
                    normalize_posterior=False,  # not necessary here
                    subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
                )
                domain_params = domain_params.squeeze(0)

                # Plot
                color_palette = sns.color_palette()[1:]
                _ = draw_posterior_scatter_2d(
                    ax=axs[ax_cnt],
                    dp_samples=[domain_params],
                    dp_mapping=algo.dp_mapping,
                    dims=(0, 1),
                    prior=prior,
                    env_sim=None,
                    env_real=algo._env_real,
                    axis_limits=None,
                    x_label="mass $m_p$",
                    y_label="length $l_p$" if idx == config["sel_rounds"][0] else None,
                    show_y_tick_labels=idx == config["sel_rounds"][0],
                    legend_labels=["npdr"],
                    show_legend=True if idx == len(posteriors) - 1 else False,
                    title=f"round {ax_cnt+1}",
                    color_palette=color_palette,
                )

            elif args.mode == "heatmap":
                # Plot
                _ = draw_posterior_heatmap_2d(
                    axs=axs[ax_cnt],
                    plot_type="joint",
                    posterior=posterior,
                    data_real=data.unsqueeze(0),
                    dp_mapping=algo.dp_mapping,
                    dims=(0, 1),
                    prior=prior,
                    env_real=algo._env_real,
                    rescale_posterior=True,
                    grid_bounds=None,
                    grid_res=200,
                    x_label="mass $m_p$",
                    y_label="length $l_p$" if idx == config["sel_rounds"][0] else None,
                    show_y_tick_labels=idx == config["sel_rounds"][0],
                    title=f"round {ax_cnt+1}",
                    contourf_kwargs=dict(cmap=cmap),
                    scatter_kwargs=dict(s=40),
                )
            else:
                raise pyrado.ValueErr(given=args.mode, eq_constraint="scatter or heatmap")

            # Increase the counter for every selected figure
            ax_cnt += 1

        # Adjust
        fig.subplots_adjust(wspace=0.05)

        if args.save:
            for fmt in ["pdf", "pgf", "png"]:
                plot_dir = os.path.join(pyrado.TEMP_DIR, "plots")
                os.makedirs(plot_dir, exist_ok=True)
                fig.savefig(
                    os.path.join(
                        plot_dir,
                        f"{algo._env_real.name}_nofrict_{config['algo_name']}_prob_iter_{args.iter}"
                        f"_rounds_{'_'.join(str(item) for item in config['sel_rounds'])}_{args.mode}_title.{fmt}",
                    ),
                    dpi=150,
                )

    if args.verbose:
        plt.show()
