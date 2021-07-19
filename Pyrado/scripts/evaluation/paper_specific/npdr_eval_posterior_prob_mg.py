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
Script to evaluate a posteriors obtained with NPDR and BayesSim in a round-wise fashion with scatter pair-plots showing
samples form the posteriors, as reported in

.. seealso::
    [1] F. Muratore, T. Gruner, F. Wiese, B. Belousov, M. Gienger, J. Peters, "TITLE", VENUE, YEAR
"""
import os

import seaborn as sns
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.environment_wrappers.utils import inner_env
from pyrado.plotting.distribution import draw_posterior_pairwise_scatter
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    plt.rc("text", usetex=args.use_tex)
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # NPDR
    ex_dir_npdr = os.path.join(pyrado.TEMP_DIR, "mg-ik", "npdr_time", "")
    algo = pyrado.load("algo.pkl", ex_dir_npdr)
    if not isinstance(algo, NPDR):
        raise pyrado.TypeErr(given=algo, expected_type=NPDR)
    env_sim = inner_env(pyrado.load("env_sim.pkl", ex_dir_npdr))
    prior_npdr = pyrado.load("prior.pt", ex_dir_npdr)
    posterior_npdr = algo.load_posterior(ex_dir_npdr, idx_iter=0, idx_round=6, obj=None, verbose=True)  # CHOICE
    data_real_npdr = pyrado.load(f"data_real.pt", ex_dir_npdr, prefix="iter_0", verbose=True)  # CHOICE
    domain_params_npdr, log_probs = SBIBase.eval_posterior(
        posterior_npdr,
        data_real_npdr,
        args.num_samples,
        normalize_posterior=False,  # not necessary here
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
    )
    domain_params_posterior_npdr = domain_params_npdr.reshape(1, -1, domain_params_npdr.shape[-1]).squeeze()

    # Bayessim
    ex_dir_bs = os.path.join(pyrado.TEMP_DIR, "mg-ik", "bayessim_time", "")
    algo = pyrado.load("algo.pkl", ex_dir_bs)
    if not isinstance(algo, BayesSim):
        raise pyrado.TypeErr(given=algo, expected_type=BayesSim)
    posterior_bs = algo.load_posterior(ex_dir_bs, idx_iter=0, idx_round=0, obj=None, verbose=True)  # CHOICE
    data_real_bs = pyrado.load(f"data_real.pt", ex_dir_bs, prefix="iter_0", verbose=True)
    domain_params_bs, log_probs = SBIBase.eval_posterior(
        posterior_bs,
        data_real_bs,
        args.num_samples,
        normalize_posterior=False,  # not necessary here
    )
    domain_params_posterior_bs = domain_params_bs.reshape(1, -1, domain_params_bs.shape[-1]).squeeze()

    # Configure
    plt.rc("text", usetex=True)
    if args.layout == "inside":
        num_rows, num_cols = len(algo.dp_mapping), len(algo.dp_mapping)
    elif args.layout == "outside":
        num_rows, num_cols = len(algo.dp_mapping) + 1, len(algo.dp_mapping) + 1
    else:
        raise NotImplementedError

    # Plot
    figsize = (6.2, 6.2)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize, tight_layout=False, constrained_layout=False)
    color_palette = sns.color_palette("tab10")
    dp_samples = [domain_params_posterior_npdr, domain_params_posterior_bs]
    legend_labels = ["NPDR", "BayesSim"]
    _ = draw_posterior_pairwise_scatter(
        axs=axs,
        dp_samples=dp_samples,
        dp_mapping=algo.dp_mapping,
        prior=prior_npdr,
        env_sim=env_sim,
        env_real=None,
        marginal_layout=args.layout,
        legend_labels=legend_labels,
        color_palette=color_palette,
        labels=[
            r"$r_b$",
            r"$m_b$",
            r"$e_b$",
            # r"$\mu_{b,dry}$",  # former: r"$\mu_{b,dry}$"
            r"$\mu_{b}$",  # former: r"$\mu_{b,roll}$"
            # r"$\Delta x_1$",
            # r"$\Delta y_1$",
            # r"$\Delta \gamma_1$",
            r"$\Delta x_2$",
            r"$\Delta y_2$",
            r"$\Delta \gamma_2$",
        ],
        prob_label=None if args.layout == "outside" else "",
        alpha=[0.6, 0.4],
        custom_scatter_args=dict(s=4),
    )

    # Configure
    for i in range(num_rows):
        for j in range(num_cols):
            axs[i, j].set_xticklabels([])
            axs[i, j].set_yticklabels([])
            axs[i, j].xaxis.labelpad = -3
            axs[i, j].yaxis.labelpad = -4
            axs[i, j].tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False)

    # Tune the gaps between the subplots by hand
    plt.subplots_adjust(hspace=0.07, wspace=0.07)  # requires tight_layout=False, constrained_layout=False

    if args.save:
        for fmt in args.save_format:
            os.makedirs(os.path.join(pyrado.TEMP_DIR, "plots"), exist_ok=True)
            rnd = f"_round_{args.round}" if args.round != -1 else ""
            fig.savefig(
                os.path.join(
                    pyrado.TEMP_DIR,
                    "plots",
                    f"mg_posterior_prob_npdr_bayessim_nom_{args.num_samples}_{figsize[0]}.{fmt}",
                ),
                dpi=300,
            )

    if args.verbose:
        plt.show()
