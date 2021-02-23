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
Script to plot dynamical systems used in Activation Dynamic Networks (ADN)
"""
import os
import numpy as np
import torch as to
import pandas as pd
import os.path as osp
from matplotlib import pyplot as plt

import pyrado
from pyrado.policies.recurrent.adn import pd_capacity_21_abs
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc("text", usetex=args.use_tex)

    # Define the configurations to plot
    pd_fcn = pd_capacity_21_abs  # function determining the potential dynamics
    p_min, p_max, num_p = -6.0, 6.0, 501
    print_cbt(f"Evaluating {pd_fcn.__name__} between {p_min} and {p_max}.", "c")

    p = to.linspace(p_min, p_max, num_p)  # endpoint included
    nominal = dict(tau=1, s=0, h=0, c=4, kappa=0.1)
    nominals = dict(
        tau=nominal["tau"] * to.ones(num_p),
        s=nominal["s"] * to.ones(num_p),
        h=nominal["h"] * to.ones(num_p),
        c=nominal["c"] * to.ones(num_p),
        kappa=nominal["kappa"] * to.ones(num_p),
    )
    config = dict()
    config["tau"] = to.linspace(0.2, 1.2, 6).repeat(num_p, 1).t()  # must include nominal value
    config["s"] = to.linspace(-2.0, 2.0, 5).repeat(num_p, 1).t()  # must include nominal value
    config["h"] = to.linspace(-1.0, 1.0, 5).repeat(num_p, 1).t()  # must include nominal value
    config["c"] = to.linspace(3.0, 5.0, 5).repeat(num_p, 1).t()  # must include nominal value
    config["kappa"] = to.linspace(0.0, 0.5, 5).repeat(num_p, 1).t()  # must include nominal value
    df_tau = pd.DataFrame(columns=["p_dot", "p", "s", "h", "tau", "c", "kappa"])
    df_s = pd.DataFrame(columns=["p_dot", "p", "s", "h", "tau", "c", "kappa"])
    df_h = pd.DataFrame(columns=["p_dot", "p", "s", "h", "tau", "c", "kappa"])
    df_c = pd.DataFrame(columns=["p_dot", "p", "s", "h", "tau", "c", "kappa"])
    df_kappa = pd.DataFrame(columns=["p_dot", "p", "s", "h", "tau", "c", "kappa"])

    if args.save:
        save_dir = osp.join(pyrado.EVAL_DIR, "dynamical_systems")
        os.makedirs(osp.dirname(save_dir), exist_ok=True)

    # Get the derivatives
    for tau in config["tau"]:
        p_dot = pd_fcn(p=p, tau=tau, s=nominals["s"], h=nominals["h"], capacity=nominals["c"], kappa=nominals["kappa"])
        df_tau = pd.concat(
            [
                df_tau,
                pd.DataFrame(
                    dict(
                        p_dot=p_dot,
                        p=p,
                        s=nominals["s"],
                        h=nominals["h"],
                        tau=tau,
                        c=nominals["c"],
                        kappa=nominals["kappa"],
                    )
                ),
            ],
            axis=0,
        )

    for s in config["s"]:
        p_dot = pd_fcn(p=p, s=s, tau=nominals["tau"], h=nominals["h"], capacity=nominals["c"], kappa=nominals["kappa"])
        df_s = pd.concat(
            [
                df_s,
                pd.DataFrame(
                    dict(
                        p_dot=p_dot,
                        p=p,
                        s=s,
                        h=nominals["h"],
                        tau=nominals["tau"],
                        c=nominals["c"],
                        kappa=nominals["kappa"],
                    )
                ),
            ],
            axis=0,
        )

    for h in config["h"]:
        p_dot = pd_fcn(p=p, h=h, s=nominals["s"], tau=nominals["tau"], capacity=nominals["c"], kappa=nominals["kappa"])
        df_h = pd.concat(
            [
                df_h,
                pd.DataFrame(
                    dict(
                        p_dot=p_dot,
                        p=p,
                        h=h,
                        s=nominals["s"],
                        tau=nominals["tau"],
                        c=nominals["c"],
                        kappa=nominals["kappa"],
                    )
                ),
            ],
            axis=0,
        )

    for c in config["c"]:
        p_dot = pd_fcn(p=p, capacity=c, s=nominals["s"], h=nominals["h"], tau=nominals["tau"], kappa=nominals["kappa"])
        df_c = pd.concat(
            [
                df_c,
                pd.DataFrame(
                    dict(
                        p_dot=p_dot,
                        p=p,
                        c=c,
                        s=nominals["s"],
                        h=nominals["h"],
                        tau=nominals["tau"],
                        kappa=nominals["kappa"],
                    )
                ),
            ],
            axis=0,
        )

    for kappa in config["kappa"]:
        p_dot = pd_fcn(p=p, kappa=kappa, s=nominals["s"], h=nominals["h"], tau=nominals["tau"], capacity=nominals["c"])
        df_kappa = pd.concat(
            [
                df_kappa,
                pd.DataFrame(
                    dict(
                        p_dot=p_dot,
                        p=p,
                        kappa=kappa,
                        s=nominals["s"],
                        h=nominals["h"],
                        tau=nominals["tau"],
                        c=nominals["c"],
                    )
                ),
            ],
            axis=0,
        )

    """ tau """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the time constant tau: s = {nominal['s']}, h = {nominal['h']}, c = {nominal['c']}, "
        f"kappa = {nominal['kappa']}"
    )
    for tau in config["tau"]:
        ax.plot(
            df_tau.loc[
                (df_tau["s"] == nominal["s"])
                & (df_tau["h"] == nominal["h"])
                & (df_tau["tau"] == tau[0].numpy())
                & (df_tau["c"] == nominal["c"])
                & (df_tau["kappa"] == nominal["kappa"])
            ]["p"],
            df_tau.loc[
                (df_tau["s"] == nominal["s"])
                & (df_tau["h"] == nominal["h"])
                & (df_tau["tau"] == tau[0].numpy())
                & (df_tau["c"] == nominal["c"])
                & (df_tau["kappa"] == nominal["kappa"])
            ]["p_dot"],
            label=f"$\\tau = {np.round(tau[0].numpy(), 2):.1f}$",
        )

    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\dot{p}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True)
    ax.grid()

    # Save
    if args.save:
        for fmt in ["pdf", "pgf"]:
            fig.savefig(osp.join(save_dir, f"potdyn-tau.{fmt}"), dpi=500)

    """ s """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the stimulus s: tau = {nominal['tau']}, h = {nominal['h']}, c = {nominal['c']},"
        f"kappa = {nominal['kappa']}"
    )
    for s in config["s"]:
        ax.plot(
            df_s.loc[
                (df_s["s"] == s[0].numpy())
                & (df_s["h"] == nominal["h"])
                & (df_s["tau"] == nominal["tau"])
                & (df_s["c"] == nominal["c"])
                & (df_s["kappa"] == nominal["kappa"])
            ]["p"],
            df_s.loc[
                (df_s["s"] == s[0].numpy())
                & (df_s["h"] == nominal["h"])
                & (df_s["tau"] == nominal["tau"])
                & (df_s["c"] == nominal["c"])
                & (df_s["kappa"] == nominal["kappa"])
            ]["p_dot"],
            label=f"$s = {np.round(s[0].numpy(), 2):.1f}$",
        )

    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\dot{p}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True)
    ax.grid()

    # Save
    if args.save:
        for fmt in ["pdf", "pgf"]:
            fig.savefig(osp.join(save_dir, f"potdyn-s.{fmt}"), dpi=500)

    """ h """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the resting level h: s = {nominal['s']}, tau = {nominal['tau']}, c = {nominal['c']},"
        f"kappa = {nominal['kappa']}"
    )
    for h in config["h"]:
        ax.plot(
            df_h.loc[
                (df_h["h"] == h[0].numpy())
                & (df_h["s"] == nominal["s"])
                & (df_h["tau"] == nominal["tau"])
                & (df_h["c"] == nominal["c"])
                & (df_h["kappa"] == nominal["kappa"])
            ]["p"],
            df_h.loc[
                (df_h["h"] == h[0].numpy())
                & (df_h["s"] == nominal["s"])
                & (df_h["tau"] == nominal["tau"])
                & (df_h["c"] == nominal["c"])
                & (df_h["kappa"] == nominal["kappa"])
            ]["p_dot"],
            label=f"$h = {np.round(h[0].numpy(), 2):.1f}$",
        )

    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\dot{p}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fancybox=True)
    ax.grid()

    """ c """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the capacity C: tau = {nominal['tau']}, s = {nominal['s']}, h = {nominal['h']}, "
        f"kappa = {nominal['kappa']}"
    )
    for c in config["c"]:
        ax.plot(
            df_c.loc[
                (df_c["c"] == c[0].numpy())
                & (df_c["h"] == nominal["h"])
                & (df_c["tau"] == nominal["tau"])
                & (df_c["s"] == nominal["s"])
                & (df_c["kappa"] == nominal["kappa"])
            ]["p"],
            df_c.loc[
                (df_c["c"] == c[0].numpy())
                & (df_c["h"] == nominal["h"])
                & (df_c["tau"] == nominal["tau"])
                & (df_c["s"] == nominal["s"])
                & (df_c["kappa"] == nominal["kappa"])
            ]["p_dot"],
            label=f"$C = {np.round(c[0].numpy(), 2):.1f}$",
        )

    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\dot{p}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True)
    ax.grid()

    # Save
    if args.save:
        for fmt in ["pdf", "pgf"]:
            fig.savefig(osp.join(save_dir, f"potdyn-C.{fmt}"), dpi=500)

    """ kappa """
    fig, ax = plt.subplots(1, figsize=(12, 10))
    fig.canvas.set_window_title(
        f"Varying the decay factor kappa: tau = {nominal['tau']}, s = {nominal['s']}, h = {nominal['h']}, "
        f"c = {nominal['c']}"
    )
    for kappa in config["kappa"]:
        ax.plot(
            df_kappa.loc[
                (df_kappa["kappa"] == kappa[0].numpy())
                & (df_kappa["tau"] == nominal["tau"])
                & (df_kappa["s"] == nominal["s"])
                & (df_kappa["h"] == nominal["h"])
                & (df_kappa["c"] == nominal["c"])
            ]["p"],
            df_kappa.loc[
                (df_kappa["kappa"] == kappa[0].numpy())
                & (df_kappa["tau"] == nominal["tau"])
                & (df_kappa["s"] == nominal["s"])
                & (df_kappa["h"] == nominal["h"])
                & (df_kappa["c"] == nominal["c"])
            ]["p_dot"],
            label=f"$\\kappa = {np.round(kappa[0].numpy(), 2):.1f}$",
        )

    ax.set_xlabel("$p$")
    ax.set_ylabel(r"$\dot{p}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fancybox=True)
    ax.grid()

    # Save
    if args.save:
        for fmt in ["pdf", "pgf"]:
            fig.savefig(osp.join(save_dir, f"potdyn-kappa.{fmt}"), dpi=500)

    plt.show()
