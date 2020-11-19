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
from torch.distributions import Normal, Uniform

import pyrado
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.algorithms.meta.simopt import SimOpt
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
    print_cbt(f'Found {num_cand} candidates of dimension {dim_cand}:\n{cands.detach().cpu().numpy()}', 'w')
    if dim_cand%2 != 0:
        raise pyrado.ShapeErr(msg='The dimension of domain distribution parameters must be a multiple of 2!')

    # Remove the initial candidates
    hparams = load_dict_from_yaml(osp.join(ex_dir, 'hyperparams.yaml'))
    if 'algo_name' not in hparams:
        raise pyrado.KeyErr(keys='algo_name', container=hparams)
    if 'dp_map' not in hparams:
        raise pyrado.KeyErr(keys='dp_map', container=hparams)

    # Process algorithms differently
    if hparams['algo_name'] == BayRn.name:
        try:
            num_init_cand = hparams['algo']['num_init_cand']
        except KeyError:
            raise KeyError('There was no num_init_cand key in the hparams.yaml file!'
                           'Are you sure you loaded a BayRn experiment?')
        if not args.load_all:
            cands = cands[num_init_cand:, :]
            # cands_values = cands_values[num_init_cand:, :]
            num_cand -= num_init_cand
            print_cbt(f'Removed the {num_init_cand} (randomly sampled) initial candidates.', 'c')
        else:
            print_cbt(f'Did not remove the {num_init_cand} (randomly sampled) initial candidates.', 'c')

    elif hparams['algo_name'] == SimOpt.name:
        pass
        # ddp_policy = to.load(osp.join(ex_dir, 'ddp_policy.pt'))
        # cands = ddp_policy.transform_to_ddp_space(cands)

    else:
        raise pyrado.ValueErr(given=hparams['algo_name'], eq_constraint=f'{BayRn.name} or {SimOpt.name}')

    # Create the figure
    fig, axs = plt.subplots(dim_cand//2,
                            figsize=(8, 12))  # 2 parameters per domain parameter for Gaussian distributions

    # Determine the evaluation grid from the means and the associated stds
    x_grid_limits = (cands[:, 0].min() - 3*cands[to.argmin(cands[:, 0]), 1],
                     cands[:, 0].max() + 3*cands[to.argmax(cands[:, 0]), 1])

    # cands = cands[[14, 1, 2, 6, 9, 10, 11, 12, 0, 13], :]
    num_cand = cands.shape[0]

    # Extract the distributions
    for idx_dp in range(dim_cand//2):  # 2 parameters per domain parameter for Gaussian or uniform distributions
        distributions = []

        for i in range(num_cand):
            dp_name, ddp_name = hparams['dp_map'][2*idx_dp + 1]  # +1 to get the second ddp for this distribution
            if ddp_name == 'std':
                distributions.append(Normal(loc=cands[i, 2*idx_dp], scale=cands[i, 2*idx_dp + 1]))
            elif ddp_name == 'halfspan':
                mean, halfspan = cands[i, 2*idx_dp], cands[i, 2*idx_dp + 1]
                distributions.append(Uniform(low=mean - halfspan, high=mean + halfspan))
            else:
                raise NotImplementedError(f'{ddp_name}')

        # Determine the evaluation grid from the means and the associated stds
        x_grid_limits = [cands[:, 2*idx_dp].min() - 3*cands[to.argmin(cands[:, 2*idx_dp]), 1],
                         cands[:, 2*idx_dp].max() + 3*cands[to.argmax(cands[:, 2*idx_dp]), 1]]
        x_grid_limits = [x.item() for x in x_grid_limits if isinstance(x, to.Tensor)]

        # Plot the distributions
        if dim_cand//2 == 1:
            fig = render_distr_evo(axs, distributions, x_grid_limits, resolution=301,
                                   x_label=f'$\\xi_{idx_dp}$', y_label=f'$p(\\xi_{idx_dp})$',
                                   distr_labels=[rf'iter {i}' for i in range(num_cand)])
        else:
            fig = render_distr_evo(axs[idx_dp], distributions, x_grid_limits, resolution=301,
                                   x_label=f'$\\xi_{idx_dp}$', y_label=f'$p(\\xi_{idx_dp})$',
                                   distr_labels=[rf'iter {i}' for i in range(num_cand)])

        if args.save_figures:
            for fmt in ['pdf', 'pgf']:
                fig.savefig(osp.join(ex_dir, f'distr_evo.{fmt}'), dpi=500)

    plt.show()
