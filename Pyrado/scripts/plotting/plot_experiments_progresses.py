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
from builtins import range
from matplotlib import pyplot as plt

import pyrado
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import read_csv_w_replace


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment()
    # ex_dir = input('Enter a root directory that contains one or more experiment directories:\n')

    # Get all sub-directories (these should contain the policy files)
    dirs = [x[0] for x in os.walk(ex_dir)][1:]

    # Collect average and best returns per iteration
    avg_returns = []
    best_returns = []

    # Plot progress of each experiment
    plt.figure(figsize=pyrado.figsize_IEEE_1col_18to10)
    for d in dirs:
        # Load the policy's data
        file = os.path.join(d, 'progress.csv')
        data = read_csv_w_replace(file)
        avg_return = data.avg_return.values
        best_return = [avg_return[0], ]

        for i in range(1, len(avg_return)):
            if avg_return[i] > best_return[i - 1]:
                best_return.append(avg_return[i])
            else:
                best_return.append(best_return[i - 1])

        avg_returns.append(avg_return)
        best_returns.append(best_return)

        plt.subplot(121)
        plt.plot(np.arange(len(avg_return)), avg_return, ls='--', lw=.8)
        plt.subplot(122)
        plt.plot(np.arange(len(best_return)), best_return, ls='--', lw=.8)

    # Plot mean return and the 1-sigma confidence interval
    plt.subplot(121)
    plt.plot(np.mean(avg_returns, axis=0), lw=2, label='Mean', color='blue', alpha=.8)
    plt.fill_between(range(100),
                     np.mean(avg_returns, axis=0) - np.std(avg_returns, axis=0),
                     np.mean(avg_returns, axis=0) + np.std(avg_returns, axis=0),
                     alpha=0.4,
                     label=r'1$\sigma$ confidence interval')

    plt.ylim(0)
    plt.xlim(0)
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')

    plt.subplot(122)
    plt.plot(np.mean(best_returns, axis=0), lw=2, label='Mean', color='blue', alpha=.8)
    plt.fill_between(range(100),
                     np.mean(best_returns, axis=0) - np.std(best_returns, axis=0),
                     np.mean(best_returns, axis=0) + np.std(best_returns, axis=0),
                     alpha=0.4,
                     label=r'1$\sigma$ confidence interval')

    plt.legend()
    plt.ylim(0)
    plt.xlim(0)
    plt.xlabel('Iteration')
    plt.ylabel('Best Average Reward')

    plt.tight_layout()
    plt.show()
