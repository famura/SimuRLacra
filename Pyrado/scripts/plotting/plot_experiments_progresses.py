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
Script for visually comparing policy learning progress over different random seeds
"""
import numpy as np
import os
import os.path as osp
import pandas as pd
from matplotlib import pyplot as plt

import pyrado
from pyrado.plotting.curve import draw_curve
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import read_csv_w_replace
from pyrado.utils.order import get_immediate_subdirs, natural_sort


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc("text", usetex=args.use_tex)

    # Get the experiments' directories to load from
    if args.dir is None:
        parent_dir = input("Please enter the parent directory for the experiments to compare:\n")
    else:
        parent_dir = args.dir
    if not osp.isdir(parent_dir):
        raise pyrado.PathErr(parent_dir)
    dirs = get_immediate_subdirs(parent_dir)
    dirs = natural_sort(dirs)

    # Collect average and best returns per iteration
    df = pd.DataFrame()
    best_returns = []

    # Plot progress of each experiment
    fig, axs = plt.subplots(2, figsize=pyrado.figsize_IEEE_1col_18to10)
    for idx, d in enumerate(dirs):
        # Load an experiment's data
        file = os.path.join(d, "progress.csv")
        data = read_csv_w_replace(file)

        # Append one column per experiment
        df = pd.concat([df, pd.DataFrame({f"ex_{idx}": data.avg_return})], axis=1)

        axs[0].plot(np.arange(len(data.avg_return)), data.avg_return, ls="--", lw=1, label=f"ex_{idx}")
        axs[0].legend()

    # Plot mean and std across columns
    draw_curve(
        "mean_std",
        axs[1],
        pd.DataFrame(dict(mean=df.mean(axis=1), std=df.std(axis=1))),
        np.arange(len(df)),
        x_label="iteration",
        y_label="average return",
    )

    plt.show()
