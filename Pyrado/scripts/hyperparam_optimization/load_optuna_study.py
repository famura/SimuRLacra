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
Load an Optuna study and print the best hyper-parameter set.
"""
import numpy as np
import optuna
import os
import os.path as osp
from matplotlib.ticker import MaxNLocator
from optuna.structs import StudyDirection
from prettyprinter import pprint

import pyrado
from matplotlib import pyplot as plt
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(show_hparams=args.show_hparams) if args.dir is None else args.dir

    # Find and load the Optuna data base
    study, study_name = None, None
    for file in os.listdir(ex_dir):
        if file.endswith(".db"):
            study_name = file[:-3]  # the file is named like the study, just need to cut the ending
            storage = f"sqlite:////{osp.join(ex_dir, file)}"
            study = optuna.load_study(study_name, storage)
            break  # assuming there is only one database

    if study is None:
        pyrado.PathErr(msg=f"No Optuna study found in {ex_dir}!")

    # Extract the values of all trials (Optuna was set to solve a minimization problem)
    trials = [t for t in study.trials if t.value is not None]  # broken trials return None
    values = np.array([t.value for t in trials])
    if study.direction == StudyDirection.MINIMIZE:
        idcs_best = values.argsort()
    else:
        idcs_best = values.argsort()[::-1]

    # Print the best parameter configurations
    print_cbt(
        f"The best parameter set of study {study_name} was found in trial_{study.best_trial.number} with value "
        f"{study.best_value} (average return on independent test rollouts).",
        "g",
        bright=True,
    )
    pprint(study.best_params, indent=4)

    for i in idcs_best[1:]:
        if not input("Print next best trial? [y / any other] ").lower() == "y":
            break
        print(f"Next best parameter set was found in trial_{i} with value {trials[i].value}")
        pprint(trials[i].params, indent=4)

    # Plot the normal histogram and with log-scaled axis
    fig, axs = plt.subplots(nrows=2, figsize=(12, 6))
    n, bins, patches = axs[0].hist(values, bins=len(trials), density=False)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    _, _, _ = axs[1].hist(values, bins=logbins, density=False)
    axs[0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0].set_xlabel("value")
    axs[1].set_xlabel("log_10 value")
    axs[1].set_xscale("log")
    axs[0].set_ylabel("count")
    axs[1].set_ylabel("count")
    axs[0].grid(True)
    axs[1].grid(True)
    plt.suptitle("Histogram of the Values (Returns)")
    plt.show()
