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
Script to plot the observations from rollouts as well as their mean and standard deviation
"""
import os
import os.path as osp

import numpy as np
import torch as to
from tabulate import tabulate

import pyrado
from pyrado.logger.experiment import ask_for_experiment, load_dict_from_yaml
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_rollouts_from_dir


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(hparam_list=args.show_hparams) if args.dir is None else args.dir

    # Load the rollouts
    rollouts, names = load_rollouts_from_dir(ex_dir)

    # load rollouts from the
    hparam, settings = None, None
    for file_name in os.listdir(ex_dir):
        if file_name.startswith("hparam") and file_name.endswith(".yaml"):
            hparam = load_dict_from_yaml(osp.join(ex_dir, file_name))
        elif file_name == "settings.yaml":
            settings = load_dict_from_yaml(osp.join(ex_dir, file_name))

    if not hparam:
        raise pyrado.PathErr(msg="No hyperparam file could be found.")

    # get the number of real rollouts from the hyperparams dict
    if hparam.get("algo_hparam", None) and hparam.get("algo_hparam").get("num_real_rollouts", None):
        num_real_rollouts = hparam.get("algo_hparam").get("num_real_rollouts", None)
    elif settings and settings.get("algo_hparam", None):
        num_real_rollouts = settings.get("algo_hparam").get("num_real_rollouts", None)
    else:
        raise pyrado.ValueErr(msg="No `num_real_rollouts` argument was found.")

    # get list of iteration numbers and sort them in ascending order
    prefix = "iter_"
    iter_idcs = [int(name[name.find(prefix) + len(prefix)]) for name in names]
    sorted_idcs = np.argsort(iter_idcs)

    # collect the rewards
    rewards = to.stack([r.undiscounted_return() for r in rollouts])
    table = []
    mean_reward = []
    std_reward = []
    for i in sorted_idcs:
        mean_reward = to.mean(rewards[i * num_real_rollouts : (i + 1) * num_real_rollouts])
        std_reward = to.std(rewards[i * num_real_rollouts : (i + 1) * num_real_rollouts])
        max_reward = to.max(rewards[i * num_real_rollouts : (i + 1) * num_real_rollouts])
        table.append([iter_idcs[i], num_real_rollouts, mean_reward, std_reward, max_reward])

    headers = ("iteration", "num real rollouts", "mean reward", "std reward", "max reward")

    # Yehaa
    print(tabulate(table, headers))

    # Save the table in a latex file if requested
    if args.save:
        # Save the table for LaTeX
        table_latex_str = tabulate(table, headers, tablefmt="latex")
        with open(osp.join(ex_dir, f"real_rollouts_rewards.tex"), "w") as tab_file:
            print(table_latex_str, file=tab_file)
