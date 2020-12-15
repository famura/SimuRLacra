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
import numpy as np
import os.path as osp
from matplotlib import pyplot as plt
from pyrado.utils.argparser import get_argparser
from warnings import warn

from pyrado.algorithms.timeseries_prediction import TSPred
from pyrado.utils.checks import check_all_equal
from pyrado.utils.experiments import load_experiment, read_csv_w_replace


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiments' directories to load from
    ex_dirs = [
        # osp.join(pyrado.EXP_DIR, 'ENV_NAME', 'ALGO_NAME', 'EX_NAME'),
        ""
    ]

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
        idx = int(input("Which data set should be selected? Enter 0-based index: "))
    else:
        idx = 0  # we only need one since they are all equal
    dataset = datasets[idx]

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

    # Plot training and testing predictions
    fig_pred, axs_pred = plt.subplots(nrows=2, figsize=(16, 10))
    fig_pred.canvas.set_window_title(dataset.name)

    for i in range(2):
        targs = dataset.data_trn_targ if i == 0 else dataset.data_tst_targ
        losses = loss_trn_list if i == 0 else loss_tst_list

        axs_pred[i].plot(targs.numpy(), label="target", c="gray", lw=1.5, ls="--")
        for idx, policy in enumerate(policies):
            preds = preds_trn_list[idx] if i == 0 else preds_tst_list[idx]
            axs_pred[i].plot(
                np.arange(num_init_samples, targs.shape[0]),
                preds.detach().numpy(),
                label=f"{policy.name} loss: {losses[idx].item():.2e}",
            )
        axs_pred[i].set_xlabel("samples")
        axs_pred[i].set_ylabel(prefix + "values")
        axs_pred[i].set_title("Training Data" if i == 0 else "Testing Data")
        axs_pred[i].legend(ncol=len(policies) + 1)

    # Plot training and testing loss
    fig_loss, axs_loss = plt.subplots(nrows=2, figsize=(16, 10))
    fig_loss.canvas.set_window_title(dataset.name)

    for i in range(2):
        for idx, policy in enumerate(policies):
            axs_loss[i].plot(logged_losses[idx][i], label=f"{policy.name}")
        axs_loss[i].set_yscale("log")
        axs_loss[i].set_xlabel("iteration")
        axs_loss[i].set_ylabel("average MSE loss")
        axs_loss[i].set_title("Training Loss" if i == 0 else "Testing Loss")
        axs_loss[i].legend(ncol=len(policies))

    plt.show()
