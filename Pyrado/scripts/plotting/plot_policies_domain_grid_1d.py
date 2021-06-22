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
Script to plot the results from the 1D domain parameter grid evaluations of multiple policies
"""
import os.path as osp

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser


REORDER = False


def q_5(x):
    # Use function definition since the function's name will later be the column name in the pivot_table
    return np.quantile(x, q=0.05)


def q_95(x):
    # Use function definition since the function's name will later be the column name in the pivot_table
    return np.quantile(x, q=0.95)


def _reorder_for_qbb_experiment(df: pd.DataFrame) -> pd.DataFrame:
    """By default the entries are ordered alphabetically. We want SPOTA, EPOpt, PPO"""
    print("Changed the order")
    return df.iloc[[2, 0, 1]]


def _plot_and_save(
    df: pd.DataFrame,
    domain_param: str,
    domain_param_label: str,
    values: str = "ret",
    nom_dp_value: float = None,
    y_lim: list = None,
    show_legend: bool = True,
    save_figure: bool = False,
    save_dir: pyrado.PathLike = None,
):
    if "policy" in df.columns and domain_param in df.columns:
        # Pivot table with multiple aggregation functions
        df_pivot = df.pivot_table(
            index="policy", columns=domain_param, values=values, aggfunc=[np.std, np.mean, q_5, q_95]
        )

        # Reorder to have a consistent plotting order in the journal
        if REORDER:
            df_pivot = _reorder_for_qbb_experiment(df_pivot)

        fig, ax = plt.subplots(1, figsize=pyrado.figsize_IEEE_1col_18to10)  # 18/10 ratio
        for index, row in df_pivot.iterrows():
            # Generate the plot
            ax.plot(df[domain_param].drop_duplicates().values, row["mean"], label=index.replace("_", r"\_"))
            ax.fill_between(
                df[domain_param].drop_duplicates().values,
                row["mean"] - 2 * row["std"],
                row["mean"] + 2 * row["std"],
                alpha=0.3,
            )
            if show_legend:
                plt.legend()
            ax.set_xlabel(domain_param_label)
            ax.set_ylabel("return")
            if y_lim is not None:
                ax.set_ylim(y_lim)

        # Plot dashed line for the nominal domain parameter value
        if nom_dp_value is not None:
            ax.axvline(nom_dp_value, c="k", ls="--", lw=1)

        # Save plot
        if save_figure:
            fig.savefig(osp.join(save_dir, f"{domain_param}_mean_std.pdf"))

        fig, ax = plt.subplots(1, figsize=pyrado.figsize_IEEE_1col_18to10)
        for index, row in df_pivot.iterrows():
            ax.plot(df[domain_param].drop_duplicates().values, row["mean"], label=index.replace("_", r"\_"))
            ax.fill_between(df[domain_param].drop_duplicates().values, row["q_5"], row["q_95"], alpha=0.3)
            if show_legend:
                plt.legend()
            ax.set_xlabel(domain_param_label)
            ax.set_ylabel("return")
            if y_lim is not None:
                ax.set_ylim(y_lim)

        # Plot dashed line for the nominal domain parameter value
        if nom_dp_value is not None:
            ax.axvline(nom_dp_value, c="k", ls="--", lw=1)

        # Save plot
        if save_figure:
            fig.savefig(osp.join(save_dir, f"{domain_param}_mean_ci.pdf"))


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc("text", usetex=args.use_tex)

    # Get the experiment's directory to load from
    eval_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the data
    df = pd.read_pickle(osp.join(eval_dir, "df_mp_grid_1d.pkl"))

    # Remove constant rows
    df = df.loc[:, df.apply(pd.Series.nunique) != 1]

    # Force pandas to treat integer values as numeric (by default they are not)
    df = df.apply(pd.to_numeric, errors="ignore")

    """ All """
    _plot_and_save(df, "g", r"$g~[\mathrm{m/s^2}]$", nom_dp_value=9.81, save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(
        df,
        "act_delay",
        r"$\Delta t_a~[\mathrm{steps}]$",
        nom_dp_value=5.0,
        show_legend=False,
        y_lim=[0, 2500],
        save_figure=args.save,
        save_dir=eval_dir,
    )

    """ QBallBalancerSim """

    _plot_and_save(
        df,
        "m_ball",
        r"$m_b~[\mathrm{kg}]$",
        nom_dp_value=0.005,
        show_legend=False,
        save_figure=args.save,
        save_dir=eval_dir,
    )

    _plot_and_save(df, "r_ball", "r$r_b$~[\mathrm{m}]", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "r_arm", r"$r_{\mathrm{arm}}~[\mathrm{m}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "l_plate", r"$l_{\mathrm{plate}~[\mathrm{m}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "J_l", r"$J_l~[\mathrm{kg m}^2]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "J_m", r"$J_m~[\mathrm{kg m}^2]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "K_g", "$K_g~[--]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "eta_g", r"$\eta_g~[--]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "eta_m", r"$\eta_m~[--]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(
        df,
        "R_m",
        r"$R_m~[\Omega]$",
        nom_dp_value=2.6,
        show_legend=False,  # QCP
        save_figure=args.save,
        save_dir=eval_dir,
    )

    _plot_and_save(df, "k_m", r"$k_m~[\mathrm{Nm/A}]$", save_figure=args.save, save_dir=eval_dir)

    # _plot_and_save(df, 'B_eq', r'$B_{\mathrm{eq}}$',
    #                save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(
        df, "c_frict", r"$c_v~[\mathrm{Ns/m}]$", nom_dp_value=0.025, save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "V_thold_x_pos", r"$V_{\mathrm{thold,x+}}~[\mathrm{V}]$", save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "V_thold_x_neg", r"$V_{\mathrm{thold,x-}}~[\mathrm{V}]$", save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "V_thold_y_pos", r"$V_{\mathrm{thold,y+}}~[\mathrm{V}]$", save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "V_thold_y_neg", r"$V_{\mathrm{thold,y-}}~[\mathrm{V}]$", save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df,
        "offset_th_x",
        r"$\mathrm{offset}_{\theta_x}~[\mathrm{deg}]$",
        save_figure=args.save,
        save_dir=eval_dir,
    )

    _plot_and_save(
        df,
        "offset_th_y",
        r"$\mathrm{offset}_{\theta_y}~[\mathrm{deg}]$",
        save_figure=args.save,
        save_dir=eval_dir,
    )

    """ QQubeSim """

    _plot_and_save(
        df, "motor_resistance", r"$R_m~[\Omega]$", nom_dp_value=8.4, save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(df, "km", r"$k_m~[\mathrm{Nm/A}]$", nom_dp_value=0.042, save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(
        df, "mass_rot_pole", r"$m_r~[\mathrm{kg}]$", nom_dp_value=0.095, save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "length_rot_pole", r"$l_r~[\mathrm{m}]$", nom_dp_value=0.085, save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "damping_rot_pole", r"$l_r~[\mathrm{Nms/rad}]$", nom_dp_value=5e-6, save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "mass_pend_pole", r"$m_p~[\mathrm{kg}]$", nom_dp_value=0.024, save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df, "length_pend_pole", r"$l_p~[\mathrm{m}]$", nom_dp_value=0.129, save_figure=args.save, save_dir=eval_dir
    )

    _plot_and_save(
        df,
        "damping_pend_pole",
        r"$l_p~[\mathrm{Nms/rad}]$",
        nom_dp_value=1e-6,
        save_figure=args.save,
        save_dir=eval_dir,
    )

    """ QCartPoleSim """

    _plot_and_save(df, "m_cart", r"$m_c~[\mathrm{kg}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "m_pole", r"$m_p~[\mathrm{kg}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "l_cart", r"$l_c~[\mathrm{m}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "l_pole", r"$l_p~[\mathrm{m}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "l_rail", r"$l_{\mathrm{rail}}~[\mathrm{m}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(
        df,
        "r_mp",
        r"$r_{\mathrm{mp}}~[\mathrm{m}]$",
        nom_dp_value=6.35e-3,
        show_legend=False,
        y_lim=[0, 2500],
        save_figure=args.save,
        save_dir=eval_dir,
    )

    # B_pole = 0.0024,  # viscous coefficient at the pole [N*s]
    # B_eq = 5.4,  # equivalent Viscous damping coefficient [N*s/m]

    _plot_and_save(df, "B_pole", r"$B_p~[\mathrm{Ns}]$", save_figure=args.save, save_dir=eval_dir)

    _plot_and_save(df, "B_eq", r"$B_{eq}~[\mathrm{Ns/m}]$", save_figure=args.save, save_dir=eval_dir)

    plt.show()
