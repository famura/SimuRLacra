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
Script to compute the DTW distance and RMSE error
"""
import os
import os.path as osp

import numpy as np
from tabulate import tabulate

import pyrado
from pyrado.sampling.bootstrapping import bootstrap_ci
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--min_exp",
        type=int,
        default=5,
        help="name of hyper-parameter file",
    )
    parser.add_argument(
        "--algo_name",
        type=str,
        default=None,
        help="name of hyper-parameter file",
    )
    args = parser.parse_args()
    if args.algo_name not in ["npsi", "snpea"]:
        raise pyrado.ValueErr(given_name=args.algo_name, eq_constraint=["npsi", "snpea"])

    algo_round = 0 if args.algo_name == "snpea" else 4

    # Get the experiment's directory to load from
    ex_dir = f"{pyrado.TEMP_DIR}/qq-su/npdr_qq-sub" if args.dir is None else args.dir

    print(f"Found the following experiments for {args.algo_name}")
    hparams_all, settings_all = {}, {}
    metric_dict = {}
    for sub_dir in [f.path for f in os.scandir(ex_dir) if f.is_dir()]:
        if sub_dir.endswith(args.algo_name.lower()):
            metric_file = None
            for file in os.listdir(sub_dir):
                if file.startswith("distance_metrics") and file.endswith(".npy"):
                    if metric_file is not None:
                        raise ImportError(f"Found more than one distance metric file for exp {sub_dir}")
                    metric_file = file
            if metric_file is None:
                print(f"No distance metric file was found for experiment {sub_dir}")
                continue

            # load numpy array
            with open(osp.join(sub_dir, metric_file), "rb") as f:
                metric_dict[sub_dir] = np.load(f)

    metric_average = {}
    for exp, metric_data in metric_dict.items():
        sort_by_index = 1  # 1: dtw_ml, 2: dtw_nom, ...
        metric_average[exp] = np.mean(metric_data[sort_by_index])

    # Find the <min_exp> best experiments for each iter by the mean value
    sorted_rewards = sorted(metric_average.values())[: args.min_exp]
    best_experiments, experiment_means = [], {}
    for k, v in metric_average.items():
        if v in sorted_rewards:
            best_experiments.append(k)
            experiment_means[k] = np.mean(metric_dict[k], axis=0)[1:]

    # Compute the metrics
    metric_arr = np.concatenate([metric_dict[exp] for exp in best_experiments])[:, 1:]
    table, conf_table = [], []
    mean_metric = np.mean(metric_arr, axis=0)
    min_metric = np.min(metric_arr, axis=0)
    std_metric = np.std(metric_arr, axis=0)
    headers = ["metric", "rollouts", "mean", "min", "std"]
    column_labels = ["dtw_ml", "dtw_nom", "rmse_ml", "rmse_nom"]
    for i in range(len(column_labels)):
        table.append([column_labels[i], metric_arr.shape[0], mean_metric[i], min_metric[i], std_metric[i]])

    print("\nAll metrics:")
    print(tabulate(metric_arr, column_labels))
    print("\nStandard Deviation:\n", tabulate(table, headers))

    # compute confidence intervals
    conf_headers = ["metric", "experiments", "mean", "ci low", "ci high"]

    for i in range(len(column_labels)):
        total_mean, total_ci_lo, total_ci_hi = bootstrap_ci(
            np.array([experiment_means[exp][i] for exp in best_experiments]),
            np.mean,
            num_reps=1000,
            alpha=0.05,
            ci_sides=2,
        )

        conf_table.append([column_labels[i], len(best_experiments), total_mean, total_ci_lo, total_ci_hi])
    print("\nConfidence Interval:\n", tabulate(conf_table, conf_headers))

    info = "Best Experiments:\n"
    for exp in best_experiments:
        info += f"\t\t{exp}:\n"

    # Save the table in a latex file if requested
    if args.save:
        # Save the table for LaTeX
        table_latex_str = tabulate(table, headers, tablefmt="latex")
        with open(osp.join(ex_dir, f"distance_metrics_{args.algo_name}.tex"), "w") as tab_file:
            print(table_latex_str, file=tab_file)
        table_latex_str_conf = tabulate(conf_table, headers, tablefmt="latex")
        with open(osp.join(ex_dir, f"distance_metrics_{args.algo_name}_confidence.tex"), "w") as conf_file:
            print(table_latex_str_conf, file=conf_file)
        with open(osp.join(ex_dir, f"distance_metrics_{args.algo_name}_info.txt"), "w") as txt_file:
            txt_file.write(info)
