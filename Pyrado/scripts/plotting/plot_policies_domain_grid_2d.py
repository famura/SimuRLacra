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
Script to plot the results from the 2D domain parameter grid evaluations of a single policy.
"""
import os
import os.path as osp
from typing import Tuple, Optional

import pandas as pd
import pyrado
from matplotlib import colors
from matplotlib import pyplot as plt
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.heatmap import draw_heatmap
from pyrado.plotting.utils import AccNorm
from pyrado.utils.argparser import get_argparser


def _plot_and_save(
    df: pd.DataFrame,
    index: str,
    column: str,
    index_label: str,
    column_label: str,
    values: str = "ret",
    add_sep_colorbar: bool = True,
    norm: colors.Normalize = None,
    nominal: Optional[Tuple[float, float]] = None,
    save_figure: bool = False,
    save_dir: str = None,
    show_figure: bool = True,
):
    if index in df.columns and column in df.columns:
        # Pivot table (by default averages over identical index / columns cells)
        df_pivot = df.pivot_table(index=[index], columns=[column], values=values)

        # Generate the plot
        fig_hm, fig_cb = draw_heatmap(
            df_pivot,
            annotate=False,
            add_sep_colorbar=add_sep_colorbar,
            norm=norm,
            y_label=index_label,
            x_label=column_label,
            add_colorbar=True,
        )
        if nominal:
            fig_hm.get_axes()[0].scatter(
                *nominal, s=150, marker="o", color="w", edgecolors="k", linewidths=4, label="Nominal"
            )
        if show_figure:
            fig_hm.show()

        # Save heat map and color bar if desired
        if save_figure:
            name = "-".join([index, column])
            fig_hm.savefig(osp.join(save_dir, f"hm-{name}.pdf"))
            if fig_cb is not None:
                fig_cb.savefig(osp.join(save_dir, f"cb-{name}.pdf"))


def _plot(dataframes, save_dirs, save_figure):
    # Commonly scale the colorbars of all plots
    accnorm = AccNorm()

    # Loop over all evaluations. Loop two times for first setting the bound of the colorbar and then saving the figures.
    for sf, show_figure in zip((False, save_figure), (False, True)):
        for df, save_dir in zip(dataframes, [None] * len(dataframes) if save_dirs is None else save_dirs):
            """ QBallBalancerSim """

            _plot_and_save(
                df,
                "m_ball",
                "r_ball",
                r"$m_{\mathrm{ball}}$",
                r"$r_{\mathrm{ball}}$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "g",
                "r_ball",
                "$g$",
                r"$r_{\mathrm{ball}}$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "J_l",
                "J_m",
                "$J_l$",
                "$J_m$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "eta_g",
                "eta_m",
                r"$\eta_g$",
                r"$\eta_m$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "k_m",
                "R_m",
                "$k_m$",
                "$R_m$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "B_eq",
                "c_frict",
                r"$B_{\mathrm{eq}}$",
                r"$c_{\mathrm{frict}}$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "V_thold_x_pos",
                "V_thold_x_neg",
                r"$V_{\mathrm{thold,x-}}$",
                r"$V_{\mathrm{thold,x+}}$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "V_thold_y_pos",
                "V_thold_y_neg",
                r"$V_{\mathrm{thold,y-}}$",
                r"$V_{\mathrm{thold,y+}}$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            _plot_and_save(
                df,
                "m_ball",
                "act_delay",
                r"$m_{\mathrm{ball}}$",
                r"$a_{\mathrm{delay}}$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                show_figure=show_figure,
            )

            """ QQubeSwingUpSim """
            _plot_and_save(
                df,
                "Dp",
                "Dr",
                r"$D_p$",
                r"$D_r$",
                add_sep_colorbar=True,
                norm=accnorm,
                save_figure=sf,
                save_dir=save_dir,
                nominal=(1e-6, 5e-6),
                show_figure=show_figure,
            )
            # _plot_and_save(
            #     df,
            #     "Mr",
            #     "Mp",
            #     r"$M_r$",
            #     r"$M_p$",
            #     add_sep_colorbar=True,
            #     norm=accnorm,
            #     save_figure=sf,
            #     save_dir=save_dir,
            #     nominal=(0.024, 0.095),
            #     show_figure=show_figure,
            # )


def _main():
    # Parse command line arguments
    argparser = get_argparser()
    argparser.add_argument(
        "--average",
        action="store_true",
        help="average over all loaded policies (default: False); create only a single heatmap",
    )
    argparser.add_argument("--save_dir", help="if --average is set, the directory to save the plot to")
    args = argparser.parse_args()

    # Get the experiment's directory to load from
    if args.dir is None:
        ex_dirs = []
        while True:
            ex_dirs.append(ask_for_experiment(show_hyper_parameters=args.show_hyperparameters, max_display=50))
            if input("Ask for more (Y/n)? ") == "n":
                break
    else:
        ex_dirs = [d.strip() for d in args.dir.split(",")]
    eval_parent_dirs = []
    for ex_dir in ex_dirs:
        eval_parent_dir = osp.join(ex_dir, "eval_domain_grid")
        if not osp.isdir(eval_parent_dir):
            raise pyrado.PathErr(given=eval_parent_dir)
        eval_parent_dirs.append(eval_parent_dir)

    if args.load_all:
        list_eval_dirs = []
        for eval_parent_dir in eval_parent_dirs:
            list_eval_dirs += [tmp[0] for tmp in os.walk(eval_parent_dir)][1:]
    else:
        list_eval_dirs = [osp.join(eval_parent_dir, "ENV_NAME", "ALGO_NAME") for eval_parent_dir in eval_parent_dirs]

    dataframes, eval_dirs = [], []
    for eval_dir in list_eval_dirs:
        assert osp.isdir(eval_dir)

        # Load the data
        pickle_file = osp.join(eval_dir, "df_sp_grid_2d.pkl")
        if not osp.isfile(pickle_file):
            print(f"{pickle_file} is not a file! Skipping...")
            continue
        df = pd.read_pickle(pickle_file)

        dataframes.append(df)
        eval_dirs.append(eval_dir)

    if args.average:
        _plot([sum(dataframes) / len(dataframes)], [args.save_dir], True)
    else:
        _plot(dataframes, eval_dirs, args.save)


if __name__ == "__main__":
    _main()
