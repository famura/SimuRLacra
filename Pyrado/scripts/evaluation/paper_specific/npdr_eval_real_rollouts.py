"""
Script to plot the observations from rollouts as well as their mean and std
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
    if hparam.get("num_real_rollouts", None) and hparam.get("num_real_rollouts", None).get("num_real_rollouts", None):
        num_real_rollouts = hparam.get("num_real_rollouts").get("num_real_rollouts", None)
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
