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
Compare potential based neural networks to classical recurrent networks for time series prediction.
"""
import os.path as osp
from warnings import warn

import numpy as np
from matplotlib import pyplot as plt

from pyrado.algorithms.regression.timeseries_prediction import TSPred
from pyrado.utils.argparser import get_argparser
from pyrado.utils.checks import check_all_equal
from pyrado.utils.experiments import load_experiment, read_csv_w_replace


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiments' directories to load from
    ex_dirs = (
        [
            # osp.join(pyrado.EXP_DIR, 'ENV_NAME', 'ALGO_NAME', 'EX_NAME'),
        ]
        if args.dir is None
        else [args.dir]
    )

    # Loading the policies data sets
    policies = []
    datasets = []
    logged_losses = []
    for ex_dir in ex_dirs:
        _, policy, kwout = load_experiment(ex_dir, args)
        policies.append(policy)
        datasets.append(kwout["dataset"])

        df = read_csv_w_replace(osp.join(ex_dir, "progress.csv"))
        logged_losses.append((df.trn_loss.values, df.tst_loss.values))

    if not check_all_equal(datasets):
        warn("Not all data sets are equal.")
        [print(d) for d in datasets]
        idx_p = int(input("Which data set should be selected? Enter 0-based index: "))
    else:
        idx_p = 0  # we only need one since they are all equal
    dataset = datasets[idx_p]

    # Adaptable settings
    num_init_samples = dataset.window_size
    windowed = False
    cascaded = False

    # Evaluate the policies on training and testing data
    preds_trn_list, loss_trn_list, preds_tst_list, loss_tst_list = [], [], [], []
    for policy in policies:
        preds_trn, loss_trn = TSPred.evaluate(
            policy, dataset.data_trn_inp, dataset.data_trn_targ, windowed, cascaded, num_init_samples=num_init_samples
        )
        preds_tst, loss_tst = TSPred.evaluate(
            policy, dataset.data_tst_inp, dataset.data_tst_targ, windowed, cascaded, num_init_samples=num_init_samples
        )
        preds_trn_list.append(preds_trn)
        loss_trn_list.append(loss_trn)
        preds_tst_list.append(preds_tst)
        loss_tst_list.append(loss_tst)

    # Prefix
    if dataset.is_standardized:
        prefix = "standardized "
    elif dataset.is_scaled:
        prefix = "scaled "
    else:
        prefix = ""

    # Create the figures
    fig_trn, axs_trn = plt.subplots(nrows=dataset.dim_data, figsize=(16, 10))
    fig_tst, axs_tst = plt.subplots(nrows=dataset.dim_data, figsize=(16, 10))
    fig_trn.canvas.set_window_title(dataset.name)
    fig_tst.canvas.set_window_title(dataset.name)

    # Plot the predictions on the training data
    for idx_dim in range(dataset.dim_data):
        axs_trn[idx_dim].plot(dataset.data_trn_targ[:, idx_dim].numpy(), lw=1.5, ls="--", c="gray", label="targ")

        for idx_p, policy in enumerate(policies):
            preds = preds_trn_list[idx_p][:, idx_dim]
            axs_trn[idx_dim].plot(
                np.arange(num_init_samples, dataset.data_trn_targ.shape[0]),
                preds.detach().cpu().numpy(),
                label=f"{policy.name} loss: {loss_trn_list[idx_p].item():.2e}",
            )

        # Annotate
        axs_trn[idx_dim].set_ylabel(prefix + "values")
        axs_trn[idx_dim].legend(ncol=len(policies) + 1)
    axs_trn[0].set_title("Training Data")
    axs_trn[-1].set_xlabel("samples")

    # Plot the predictions on the testing data
    for idx_dim in range(dataset.dim_data):
        axs_tst[idx_dim].plot(dataset.data_tst_targ[:, idx_dim].numpy(), lw=1.5, ls="--", c="gray", label="targ")

        for idx_p, policy in enumerate(policies):
            preds = preds_tst_list[idx_p][:, idx_dim]
            axs_tst[idx_dim].plot(
                np.arange(num_init_samples, dataset.data_tst_targ.shape[0]),
                preds.detach().cpu().numpy(),
                label=f"{policy.name} loss: {loss_tst_list[idx_p].item():.2e}",
            )

        # Annotate
        axs_tst[idx_dim].set_ylabel(prefix + "values")
        axs_tst[idx_dim].legend(ncol=len(policies) + 1)
    axs_tst[0].set_title("Testing Data")
    axs_tst[-1].set_xlabel("samples")

    # Plot training and testing loss
    fig_loss, axs_loss = plt.subplots(nrows=2, figsize=(16, 10))
    fig_loss.canvas.set_window_title(dataset.name)

    for idx_dim in range(2):
        for idx_p, policy in enumerate(policies):
            axs_loss[idx_dim].plot(logged_losses[idx_p][idx_dim], label=f"{policy.name}")
        axs_loss[idx_dim].set_yscale("log")
        axs_loss[idx_dim].set_xlabel("iteration")
        axs_loss[idx_dim].set_ylabel("average MSE loss")
        axs_loss[idx_dim].set_title("Training Loss" if idx_dim == 0 else "Testing Loss")
        axs_loss[idx_dim].legend(ncol=len(policies))

    plt.show()
