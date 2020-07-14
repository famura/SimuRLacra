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
Script to visualize a selection of experiments (used for benchmarking BayRn) as a box plot or violin plot
"""
import os
import os.path as osp
import numpy as np
import torch as to
from collections import OrderedDict

import pyrado
from matplotlib import pyplot as plt
from pyrado.plotting.categorial import render_boxplot, render_violinplot
from pyrado.utils.argparser import get_argparser


def _label_from_key(k: str) -> str:
    """ Helper function to create the x-tick labels from the keys in the result dict """
    if k == 'bayrn':
        return 'BayRn'
    elif k == 'epopt':
        return 'EPOpt'
    else:
        return k.upper()


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the the experiments directory
    env_name = 'qq'
    ex_name = 'bayrn_clrs'  # 'bayrn_nom_sel'
    ex_dir = osp.join(pyrado.EVAL_DIR, f'{env_name}_experiments', ex_name)

    # Use an ordered dict to store and filter the results
    result_dict = OrderedDict(
        bayrn=[],
        # epopt=[],
        udr=[],
        ppo=[]
    )

    # Get all directories for the experiments
    all_files = [tmp[2] for tmp in os.walk(ex_dir)][0]  # [tmp[0] for tmp in os.walk(ex_dir)][1:]

    for key, _ in result_dict.items():
        matches = list(filter(lambda d: key in d, all_files))
        # matches = list(filter(lambda d: key in d and '--good' not in d, all_files))

        for m in matches:
            # Load the array or tensor containing the returns of the current match i.e. policy
            if m.endswith('.npy'):
                rets = np.load(osp.join(ex_dir, m))
            elif m.endswith('.pt'):
                rets = to.load(osp.join(ex_dir, m)).numpy()
            else:
                raise FileNotFoundError
            result_dict[key].extend(rets)
        print(f'Loaded {len(result_dict[key])} data points for {key}.')

    # Extract the data
    data = [result_dict[key] for key in result_dict.keys()]  # keys() preserves irder
    x_labels = [_label_from_key(k) for k in result_dict.keys()]

    # Plot and save
    fig_fize = pyrado.figsize_IEEE_1col_18to10
    fig, ax = plt.subplots(1, figsize=fig_fize, tight_layout=True)
    means = [f'{k}: {np.mean(v):.1f}' for k, v in result_dict.items()]
    fig.canvas.set_window_title(f'Mean returns on real {env_name.upper()} -- ' + ' '.join(means))

    if args.mode == 'box':
        render_boxplot(ax, data, x_labels, y_label='return', vline_level=400,
                       show_legend=False, show_fliers=False)
    elif args.mode == 'violin':
        render_violinplot(ax, data, x_labels, y_label='return', vline_level=400,
                          show_legend=False, show_inner_quartiles=False, use_seaborn=True)
    else:
        raise pyrado.ValueErr(given=args.mode, eq_constraint='The mode must be either box or violin!')

    # Save and show
    if args.save_figures:
        for fmt in ['pdf', 'pgf']:
            fig.savefig(osp.join(ex_dir, f'returns-{args.mode}plot.{fmt}'), dpi=500)
    plt.show()
