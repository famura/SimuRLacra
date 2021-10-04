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
Script to evaluate the UCBOG and OG of multiple experiments
"""
import os
import os.path as osp

import pandas as pd
from matplotlib import pyplot as plt

import pyrado
from pyrado.logger.experiment import load_dict_from_yaml
from pyrado.sampling.sequences import *
from pyrado.utils.ordering import filter_los_by_lok


if __name__ == "__main__":
    save_name = "FILL_IN"

    # Add every experiment with (partially) matching key
    filter_key = ["FILL_IN"]

    # Get the experiments' directories to load from
    ex_dirs = []
    ex_dirs.extend([tmp[0] for tmp in os.walk(osp.join(pyrado.EXP_DIR, "ENV_NAME", "ALGO_NAME"))][1:])
    ex_dirs = filter_los_by_lok(ex_dirs, filter_key)
    print(f"Number of loaded experiments: {len(ex_dirs)}")

    dfs = []
    for ex_dir in ex_dirs:
        dfs.append(pd.read_csv(osp.join(ex_dir, "OG_log.csv")))
    df = pd.concat(dfs, axis=0)  # missing values are filled with nan

    # Compute metrics using pandas (nan values are ignored)
    print(f"Index counts\n{df.index.value_counts()}")

    # Compute metrics using pandas (nan values are ignored)
    UCBOG_mean = df.groupby(df.index)["UCBOG"].mean()
    UCBOG_std = df.groupby(df.index)["UCBOG"].std()
    Gn_mean = df.groupby(df.index)["Gn_est"].mean()
    Gn_std = df.groupby(df.index)["Gn_est"].std()
    rnd_mean = df.groupby(df.index)["ratio_neg_diffs"].mean()
    rnd_std = df.groupby(df.index)["ratio_neg_diffs"].std()

    # Reconstruct the number of domains per iteration
    nc = []
    for ex_dir in ex_dirs:
        hparam = load_dict_from_yaml(osp.join(ex_dir, "hyperparams.yaml"))
        nc_init = hparam["SPOTA"]["nc_init"]
        if hparam["SPOTA"]["sequence_cand"] == "sequence_add_init":
            nc.append(sequence_add_init(nc_init, len(UCBOG_mean) - 1)[1])
        elif hparam["SPOTA"]["sequence_cand"] == "sequence_rec_double":
            nc.append(sequence_rec_double(nc_init, len(UCBOG_mean) - 1)[1])
        elif hparam["SPOTA"]["sequence_cand"] == "sequence_rec_sqrt":
            nc.append(sequence_rec_sqrt(nc_init, len(UCBOG_mean) - 1)[1])
        else:
            raise pyrado.ValueErr(
                given=hparam["SPOTA"]["sequence_cand"],
                eq_constraint="'sequence_add_init', 'sequence_rec_double', 'sequence_rec_sqrt'",
            )
    nc_means = np.floor(np.mean(np.asarray(nc), axis=0))

    # Plots
    fig1, axs = plt.subplots(3, constrained_layout=True)
    axs[0].plot(UCBOG_mean.index, UCBOG_mean, label="UCBOG")
    axs[0].fill_between(UCBOG_mean.index, UCBOG_mean - 2 * UCBOG_std, UCBOG_mean + 2 * UCBOG_std, alpha=0.3)
    axs[1].plot(Gn_mean.index, Gn_mean, label=r"$\hat{G}_n$")
    axs[1].fill_between(Gn_mean.index, Gn_mean - 2 * Gn_std, Gn_mean + 2 * Gn_std, alpha=0.3)
    axs[2].plot(rnd_mean.index, rnd_mean, label="rnd")
    axs[2].fill_between(rnd_mean.index, rnd_mean - 2 * rnd_std, rnd_mean + 2 * rnd_std, alpha=0.3)

    fig2, ax1 = plt.subplots(1, figsize=pyrado.figsize_IEEE_1col_18to10)
    fig2.canvas.manager.set_window_title(f"Final UCBOG value: {UCBOG_mean.values[-1]}")
    ax1.plot(UCBOG_mean.index, UCBOG_mean, label="UCBOG")
    ax1.fill_between(UCBOG_mean.index, UCBOG_mean - 2 * UCBOG_std, UCBOG_mean + 2 * UCBOG_std, alpha=0.3)
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("UCBOG", color="C0")
    ax2 = ax1.twinx()  # second y axis
    ax2.plot(nc_means, color="C1")
    ax2.set_ylabel("number of domains $n_c$", color="C1")
    fig2.savefig(osp.join(pyrado.EVAL_DIR, "optimality_gap", "ucbog_" + save_name + ".pdf"), dpi=500)
    plt.show()
