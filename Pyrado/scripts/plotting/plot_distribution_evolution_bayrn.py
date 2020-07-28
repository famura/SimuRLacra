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
Script to plot the evolution of the domain parameter distribution after a Bayesian Domain Adaptation experiment
"""
import os.path as osp
import torch as to
from matplotlib import pyplot as plt
from torch.distributions import Normal

import pyrado
from pyrado.logger.experiment import ask_for_experiment, load_dict_from_yaml
from pyrado.plotting.distribution import render_distr_evo
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc('text', usetex=args.use_tex)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.ex_dir is None else args.ex_dir

    # Load the data
    cands = to.load(osp.join(ex_dir, 'candidates.pt'))
    # cands_values = to.load(osp.join(ex_dir, 'candidates_values.pt')).unsqueeze(1)
    num_cand = cands.shape[0]  # number of samples i.e. iterations of BayRn (including init phase)
    dim_cand = cands.shape[1]  # number of domain distribution parameters
    if dim_cand%2 != 0:
        raise pyrado.ValueErr(msg='The dimension of domain distribution parameters must be a multiple of 2!')

    # Remove the initial candidates
    hparams = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
    try:
        num_init_cand = hparams['BayRn']['num_init_cand']
    except KeyError:
        raise KeyError('There was no BayRn or num_init_cand key in the hyperparameters.yaml file!'
                       'Are you sure you loaded a BayRn experiment?')

    if not args.load_all:
        cands = cands[num_init_cand:, :]
        # cands_values = cands_values[num_init_cand:, :]
        num_cand -= num_init_cand
        print_cbt(f'Removed the {num_init_cand} (randomly sampled) initial candidates.', 'y')
    else:
        print_cbt(f'Did not remove the {num_init_cand} (randomly sampled) initial candidates.', 'y')

    # Create the figure
    fig, axs = plt.subplots(dim_cand//2)  # 2 parameters per domain parameter for Gaussian distributions

    # Determine the evaluation grid from the means and the associated stds
    x_grid_limits = (cands[:, 0].min() - 3*cands[to.argmin(cands[:, 0]), 1],
                     cands[:, 0].max() + 3*cands[to.argmax(cands[:, 0]), 1])

    cands = cands[[14, 1, 2, 6, 9, 10, 11, 12, 0, 13], :]
    num_cand = cands.shape[0]

    # Extract the distributions
    for dp in range(dim_cand//2):  # 2 parameters per domain parameter for Gaussian distributions
        distributions = []

        for i in range(num_cand):
            distributions.append(Normal(loc=cands[i, dp*2], scale=cands[i, dp*2 + 1]))

        # Determine the evaluation grid from the means and the associated stds
        x_grid_limits = (cands[:, dp*2].min() - 3*cands[to.argmin(cands[:, dp*2]), 1],
                         cands[:, dp*2].max() + 3*cands[to.argmax(cands[:, dp*2]), 1])

        # Plot the normal distributions
        if dim_cand//2 == 1:
            fig = render_distr_evo(axs, distributions, x_grid_limits,
                                   x_label=f'$\\xi_{dp}$', y_label=f'$p(\\xi_{dp})$',
                                   distr_labels=[f'iter\_{i}' for i in range(num_cand)])
        else:
            fig = render_distr_evo(axs[dp], distributions, x_grid_limits,
                                   x_label=f'$\\xi_{dp}$', y_label=f'$p(\\xi_{dp})$',
                                   distr_labels=[f'iter\_{i}' for i in range(num_cand)])

        if args.save_figures:
            for fmt in ['pdf', 'pgf']:
                fig.savefig(osp.join(ex_dir, f'distribution-evolution.{fmt}'), dpi=500)

    plt.show()
