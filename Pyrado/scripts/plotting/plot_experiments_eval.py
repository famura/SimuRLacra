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
Script to visualize the (real-world) return of a selection of experiments as a box plot or violin plot
"""
import os
import os.path as osp
import numpy as np
import torch as to
from tabulate import tabulate

import pyrado
from matplotlib import pyplot as plt
from pyrado.plotting.categorical import draw_categorical
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt
from pyrado.utils.order import get_immediate_subdirs, natural_sort


class ResultContainer:
    """
    Extract the returns from all experiments in a given directory. The evaluations are expected to be in a sub-folder
    of the experiments called `evaluation`. If there are multiple evaluation files, the returns are averaged.
    """

    def __init__(self,
                 name: str,
                 parent_dir: str,
                 incl_pattern: str = None,
                 excl_pattern: str = None,
                 latest_evals_only: bool = False,
                 eval_subdir_name: str = 'evaluation',
                 sort: bool = False):
        """
        Constructor

        :param name: label for the data, e.g. name of the algorithm
        :param parent_dir: path to the algorithm's directory
        :param incl_pattern: only include experiments if their names partially contain the include pattern
        :param excl_pattern: exclude experiments if their names do not even partially contain the exclude pattern
        :param latest_evals_only: if `True` only the very latest evaluation file is loaded to estimate the returns
        :param sort: sort the found experiments by name, i.e. by date
        """
        if not osp.isdir(parent_dir):
            raise pyrado.PathErr(given=parent_dir)
        if incl_pattern is not None and not isinstance(incl_pattern, str):
            raise pyrado.TypeErr(given=incl_pattern, expected_type=str)
        if excl_pattern is not None and not isinstance(excl_pattern, str):
            raise pyrado.TypeErr(given=excl_pattern, expected_type=str)

        self.name = name
        self.parent_dir = parent_dir
        self.incl_pattern = incl_pattern
        self.excl_pattern = excl_pattern
        self.latest_evals_only = latest_evals_only
        self.eval_subdir_name = eval_subdir_name

        # Include experiments
        self.matches = get_immediate_subdirs(parent_dir)
        if sort:
            self.matches = natural_sort(self.matches)

        if self.incl_pattern is not None:
            # Only include experiments if their names partially contain the include pattern
            self.matches = list(filter(lambda d: self.incl_pattern in d, self.matches))

        if self.excl_pattern is not None:
            # Exclude experiments if their names do not even partially contain the exclude pattern
            self.matches = list(filter(lambda d: self.excl_pattern not in d, self.matches))

        self._returns_est_per_ex = []
        self.returns_est = []
        cnt_nonexist_dirs = 0
        for match in self.matches:
            # Get the evaluation subdirectory
            eval_dir = osp.join(match, self.eval_subdir_name)

            if osp.exists(eval_dir):
                # Crawl through the experiment's evaluation directory
                rets = []  # empirical returns from the experiments
                num_samples = []  # number of samples per return estimate
                for root, dirs, files in os.walk(eval_dir):
                    files.sort(reverse=True)  # in case there are multiple evaluations
                    for f in files:
                        if f.endswith('.npy'):
                            rets.append(np.load(osp.join(eval_dir, f)))
                            num_samples.append(len(rets))
                        elif f.endswith('.pt'):
                            rets.append(to.load(osp.join(eval_dir, f)).numpy())
                        else:
                            raise FileNotFoundError

                        # Only include the latest evaluation found in the folder
                        if self.latest_evals_only:
                            break

            else:
                cnt_nonexist_dirs += 1

            # Store the estimated return per evaluation run (averaged over individual evaluations)
            self._returns_est_per_ex.append(np.mean(np.asarray(rets), axis=1))
            self.returns_est.extend(np.mean(np.asarray(rets), axis=1))

        # Print what has been loaded
        ex_names = ['...' + m[m.rfind('/'):] for m in self.matches]  # cut off everything until the experiment's name
        print(tabulate(
            [[ex_name, ret] for ex_name, ret in zip(ex_names, self._returns_est_per_ex)],
            headers=['Loaded directory', 'Returns averaged per experiment']
        ))

        if cnt_nonexist_dirs == 0:
            print_cbt('All evaluation sub-directories have been found.', 'g')
        else:
            print_cbt(f'{cnt_nonexist_dirs} evaluation sub-directories have been missed.', 'y')


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the the experiments directory
    env_name = 'qq-su' if args.env_name is None else args.env_name

    # Extract the return values
    results = [
        ResultContainer(
            name='BayRn',
            parent_dir=osp.join(pyrado.EXP_DIR, env_name, 'DIR_NAME'),
            # incl_pattern='',
            latest_evals_only=False,
            sort=True
        ),
        ResultContainer(
            name='SimOpt',
            parent_dir=osp.join(pyrado.EXP_DIR, env_name, 'DIR_NAME'),
            # incl_pattern='',
            latest_evals_only=False,
            sort=True
        ),
        ResultContainer(
            name='UDR',
            parent_dir=osp.join(pyrado.EXP_DIR, env_name, 'DIR_NAME'),
            # incl_pattern='',
            latest_evals_only=False,
            sort=True
        ),
        ResultContainer(
            name='PPO',
            parent_dir=osp.join(pyrado.EXP_DIR, env_name, 'DIR_NAME'),
            # incl_pattern='',
            latest_evals_only=False,
            sort=True
        ),
    ]

    # Extract the data
    data = [r.returns_est for r in results]
    algo_names = [r.name for r in results]

    # Plot and save
    fig_fize = (3.5, 2.5/18*10)  # pyrado.figsize_IEEE_1col_18to10 = (3.5, 3.5/18*10)
    fig, ax = plt.subplots(1, figsize=fig_fize, constrained_layout=True)
    means_str = [f'{k}: {np.mean(v)}' for k, v in zip(algo_names, data)]
    fig.canvas.set_window_title(f'Mean returns on real {env_name.upper()}: ' + ', '.join(means_str))

    draw_categorical(ax, data, args.mode, x_label=algo_names, y_label='return', show_legend=False,
                     vline_level=375, vline_label='approx.\nsolved', plot_kwargs=None)

    # Save and show
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(pyrado.TEMP_DIR, f'returns_{env_name}_{args.mode}plot.{fmt}'), dpi=500, backend='pgf')
    plt.show()
