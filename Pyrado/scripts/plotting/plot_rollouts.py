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
Script to plot the observations from rollouts as well as their mean and std
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.curve import draw_curve
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_rollouts_from_dir


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc("text", usetex=False)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the rollouts
    rollouts = load_rollouts_from_dir(ex_dir)

    # Extract observations
    data = pd.DataFrame()
    for ro in rollouts:
        ro.numpy()
        df = pd.DataFrame(ro.observations, columns=ro.rollout_info["env_spec"].obs_space.labels)
        data = pd.concat([data, df], axis=1)
    means = data.groupby(by=data.columns, axis=1).mean()
    stds = data.groupby(by=data.columns, axis=1).std()

    dim_obs = rollouts[0].observations.shape[1]  # assuming same for all rollouts
    num_rows, num_cols = num_rows_cols_from_length(dim_obs, transposed=True)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 9), tight_layout=True)

    for idx_o, c in enumerate(data.columns.unique()):
        draw_curve(
            "mean_std",
            axs[idx_o // num_cols, idx_o % num_cols],
            pd.DataFrame(dict(mean=means[c], std=stds[c])),
            np.arange(len(data)),
            show_legend=False,
            x_label="steps",
            y_label=str(c),
        )

    plt.show()
