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

import joblib as jl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.utils import AccNorm
from pyrado.utils.argparser import get_argparser


def q_5(x):
    # Use function definition since the function's name will later be the column name in the pivot_table
    return np.quantile(x, q=0.05)


def q_95(x):
    # Use function definition since the function's name will later be the column name in the pivot_table
    return np.quantile(x, q=0.95)


def _plot_and_save(
    df: pd.DataFrame,
    index: str,
    index_label: str,
    value: str = "ret",
    value_label: str = "Return",
    nom_dp_value: float = None,
    y_lim: list = None,
    show_legend: bool = True,
    save_figure: bool = False,
    save_dir: str = None,
):
    if index in df.columns and value in df.columns:
        df_grouped = df.groupby(index)[value].agg([np.mean, np.std, q_5, q_95])

        # Create plot with standard deviation as shaded region.
        # fig, ax = plt.subplots(figsize=pyrado.figsize_IEEE_1col_18to10)
        fig, ax = plt.subplots()
        ax.plot(df_grouped.index, df_grouped["mean"], label="Mean")
        ax.fill_between(
            df_grouped.index,
            df_grouped["mean"] - 2 * df_grouped["std"],
            df_grouped["mean"] + 2 * df_grouped["std"],
            alpha=0.3,
            label="95% Standard Deviation",
        )
        if show_legend:
            ax.legend()
        ax.set_xlabel(index_label)
        ax.set_ylabel(value_label)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        # Plot dashed line for the nominal domain parameter value
        if nom_dp_value is not None:
            ax.axvline(nom_dp_value, c="k", ls="--", lw=1)
        # Save plot
        if save_figure:
            fig.savefig(osp.join(save_dir, f"{value}_mean_std.pdf"))

        # Create plot with quantiles as shaded region.
        # fig, ax = plt.subplots(figsize=pyrado.figsize_IEEE_1col_18to10)
        fig, ax = plt.subplots()
        ax.plot(df_grouped.index, df_grouped["mean"], label="Mean")
        ax.fill_between(df_grouped.index, df_grouped["q_5"], df_grouped["q_95"], alpha=0.3, label="95% Quantiles")
        if show_legend:
            ax.legend()
        ax.set_xlabel(index_label)
        ax.set_ylabel(value_label)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        # Plot dashed line for the nominal domain parameter value
        if nom_dp_value is not None:
            ax.axvline(nom_dp_value, c="k", ls="--", lw=1)
        # Save plot
        if save_figure:
            fig.savefig(osp.join(save_dir, f"{value}_mean_ci.pdf"))


def plot_policy(args, ex_dir):
    plt.rc("text", usetex=args.use_tex)

    # Get the experiment's directory to load from
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
        pickle_file = osp.join(eval_dir, "df_sp_grid_1d.pkl")
        if not osp.isfile(pickle_file):
            print(f"{pickle_file} is not a file! Skipping...")
            continue
        df = jl.load(pickle_file)

        # Remove constant rows
        df = df.loc[:, df.apply(pd.Series.nunique) != 1]

        _plot_and_save(
            df,
            "g",
            r"$g$",
            nom_dp_value=9.81,
            save_figure=args.save,
            save_dir=eval_dir,
        )

    plt.show()


if __name__ == "__main__":
    # Parse command line arguments
    g_args = get_argparser().parse_args()
    if g_args.load_all and g_args.dir:
        if not os.path.isdir(g_args.dir):
            raise pyrado.PathErr(given=g_args.dir)

        g_ex_dirs = [tmp[0] for tmp in os.walk(g_args.dir) if "policy.pt" in tmp[2]]
    elif g_args.dir is None:
        g_ex_dirs = [ask_for_experiment(hparam_list=g_args.show_hparams, max_display=50)]
    else:
        g_ex_dirs = [g_args.dir]

    print(f"Plotting all of {g_ex_dirs}.")
    for g_ex_dir in g_ex_dirs:
        print(f"Plotting {g_ex_dir}.")
        plot_policy(g_args, g_ex_dir)
