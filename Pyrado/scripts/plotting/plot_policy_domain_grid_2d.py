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
import pandas as pd
from matplotlib import colors
from matplotlib import pyplot as plt
from typing import Tuple, Optional

import pyrado
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
    save_dir: pyrado.PathLike = None,
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
            fig_hm.get_axes()[0].scatter(*nominal, s=100, marker="*", color="tab:green", label="")

        # Save heat map and color bar if desired
        if save_figure:
            name = "-".join([index, column])
            for fmt in ["pdf"]:
                fig_hm.savefig(osp.join(save_dir, f"hm-{name}.{fmt}"))
                if fig_cb is not None:
                    fig_cb.savefig(osp.join(save_dir, f"cb-{name}.{fmt}"))


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc("text", usetex=args.use_tex)

    # Commonly scale the colorbars of all plots
    accnorm = AccNorm()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir
    eval_parent_dir = osp.join(ex_dir, "eval_domain_grid")
    if not osp.isdir(eval_parent_dir):
        raise pyrado.PathErr(given=eval_parent_dir)

    if args.load_all:
        list_eval_dirs = [tmp[0] for tmp in os.walk(eval_parent_dir)][1:]
    else:
        list_eval_dirs = [
            osp.join(eval_parent_dir, "ENV_NAME", "ALGO_NAME"),
        ]

    # Loop over all evaluations
    for eval_dir in list_eval_dirs:
        assert osp.isdir(eval_dir)

        # Load the data
        df = pd.read_pickle(osp.join(eval_dir, "df_sp_grid_nd.pkl"))

        # Remove constant rows
        # df = df.loc[:, df.apply(pd.Series.nunique) != 1]

        """ QBallBalancerSim """

        _plot_and_save(
            df,
            "m_ball",
            "r_ball",
            r"$m_{\mathrm{ball}}$",
            r"$r_{\mathrm{ball}}$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "g",
            "r_ball",
            "$g$",
            r"$r_{\mathrm{ball}}$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "J_l",
            "J_m",
            "$J_l$",
            "$J_m$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "eta_g",
            "eta_m",
            r"$\eta_g$",
            r"$\eta_m$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "k_m",
            "R_m",
            "$k_m$",
            "$R_m$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "B_eq",
            "c_frict",
            r"$B_{\mathrm{eq}}$",
            r"$c_{\mathrm{frict}}$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "V_thold_x_pos",
            "V_thold_x_neg",
            r"$V_{\mathrm{thold,x-}}$",
            r"$V_{\mathrm{thold,x+}}$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "V_thold_y_pos",
            "V_thold_y_neg",
            r"$V_{\mathrm{thold,y-}}$",
            r"$V_{\mathrm{thold,y+}}$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
        )

        _plot_and_save(
            df,
            "m_ball",
            "act_delay",
            r"$m_{\mathrm{ball}}$",
            r"$a_{\mathrm{delay}}$",
            add_sep_colorbar=True,
            norm=accnorm,
            save_figure=args.save,
            save_dir=eval_dir,
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
            save_figure=args.save,
            save_dir=eval_dir,
            nominal=(1e-6, 5e-6),
        )

    plt.show()
