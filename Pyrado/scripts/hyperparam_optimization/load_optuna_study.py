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
from prettyprinter import pprint

from pyrado.logger.experiment import ask_for_experiment
from matplotlib import pyplot as plt
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()

    # Find and load the Optuna data base
    study, study_name = None, None
    for file in os.listdir(ex_dir):
        if file.endswith('.db'):
            study_name = file[:-3]  # we named the file like the study
            storage = f'sqlite:////{osp.join(ex_dir, file)}'
            study = optuna.load_study(study_name, storage)
            break  # assuming there is only one database

    if study is None:
        print_cbt('No study found!', 'r', bright=True)

    # Extract the values of all trials (optuna was set to solve a minimization problem)
    values = np.array([t.value for t in study.trials])
    values = -1*values[values != np.array(None)]  # broken trials return None

    # Print the best parameter configuration
    print_cbt(f'Best parameter set (trial_{study.best_trial.number}) from study {study_name} with average return '
              f'{-study.best_value}', 'g', bright=True)
    pprint(study.best_params, indent=4)

    # Plot a histogram
    fig, ax = plt.subplots(1, figsize=(8, 6))
    n, bins, patches = plt.hist(values, len(study.trials), density=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title('Histogram of the Returns')
    plt.xlabel('return')
    plt.ylabel('count')
    plt.grid(True)
    plt.show()
