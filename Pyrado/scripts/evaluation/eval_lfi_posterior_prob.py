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
Script to evaluate a posterior obtained using the sbi package.
By default (args.iter = -1), the most recent iteration is evaluated.
"""
import os
import torch as to
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.inference.lfi import NPDR
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.distribution import draw_posterior_distr_2d, draw_posterior_distr_pairwise, draw_posterior_distr_1d
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt
from pyrado.utils.order import natural_sort


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--layout",
        default="outside",
        help="plotting layout for pairplot, possibilities: inside and outside (default: inside)",
    )
    args = parser.parse_args()
    plt.rc("text", usetex=args.use_tex)
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment(show_hparams=args.show_hparams) if args.dir is None else args.dir

    # Load the algorithm
    algo = Algorithm.load_snapshot(ex_dir)
    if not (isinstance(algo, NPDR) or isinstance(algo, BayesSim)):
        raise pyrado.TypeErr(given=algo, expected_type=NPDR)

    # Load the environments, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)
    env_real = pyrado.load(None, "env_real", "pkl", ex_dir)
    prior = kwout["prior"]
    posterior = kwout["posterior"]
    observations_real = kwout["observations_real"]

    # Load the sequence of posteriors if desired
    if args.mode.lower() == "evolution":
        # Repeat the data for every posterior per round
        observations_real = to.repeat_interleave(to.atleast_2d(observations_real), algo.num_sbi_rounds, dim=0)

        # Crawl through the experiment's directory
        for root, dirs, files in os.walk(ex_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            files = natural_sort(files)
            posterior = [
                pyrado.load(None, name=f[: f.rfind(".")], file_ext=f[f.rfind(".") + 1 :], load_dir=root)
                for f in files
                if f.startswith("iter_") and f.endswith("_posterior.pt") and "round" in f
            ]
            if not posterior:
                raise pyrado.ValueErr(
                    msg="No posterior found! Most likely they haven#t been saved every round. "
                    "Use a different plotting mode."
                )
        # Repeat if there have been multiple real rollouts per iteration
        posterior = [p for p in posterior for _ in range(algo.num_real_rollouts)]

    if args.mode.lower() == "evolution" and observations_real.shape[0] > len(posterior):
        print_cbt(
            f"Found {observations_real.shape[0]} observations but {len(posterior)} posteriors, thus truncated "
            f"the superfluous observations.",
            "y",
        )
        observations_real = observations_real[: len(posterior), :]

    # Select the domain parameters to plot
    if len(algo.dp_mapping) == 1:
        idcs_dp = 0
    elif len(algo.dp_mapping) == 2:
        idcs_dp = (0, 1)
    elif args.idcs is not None:
        idcs_dp = args.idcs
    elif args.mode.lower() != "pairwise":
        usr_inp = input(
            f"Found the domain parameter mapping {algo.dp_mapping}. Select 1 or 2 domain parameter by index "
            f"to be plotted (format: separated by a whitespace):\n"
        )
        idcs_dp = tuple(map(int, usr_inp.split()))
    else:
        # We are using all dims for pairwise plot. We only set this here to jump over the len(idcs_dp) == 1 case later.
        idcs_dp = tuple(i for i in algo.dp_mapping.keys())

    # Set the condition if necessary
    if (2 >= len(algo.dp_mapping) == len(idcs_dp)) and args.mode.lower() != "pairwise":
        # No condition necessary since dim(posterior) = dim(grid)
        condition = None
    else:
        # Use the latest posterior to sample domain parameters to obtain a condition
        domain_params, log_probs = NPDR.eval_posterior(
            posterior[-1] if args.mode.lower() == "evolution" else posterior,
            observations_real,
            args.num_samples,
            normalize_posterior=False,  # not necessary here
            sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
        )
        condition = to.mean(domain_params[:, to.argmax(log_probs, dim=1), :], dim=[0, 1])
        # condition = to.mean(domain_params, dim=[0, 1])

    # Plot the posterior distribution, the true parameters / their distribution
    if len(idcs_dp) == 1:
        fig, axs = plt.subplots(figsize=(14, 7), tight_layout=True)
        _ = draw_posterior_distr_1d(
            axs,
            posterior[-1] if args.mode.lower() == "evolution" else posterior,  # ignore plotting mode
            observations_real,
            algo.dp_mapping,
            idcs_dp,
            prior,
            env_real,
            condition,
            normalize_posterior=args.normalize,
            rescale_posterior=args.rescale,
            # x_label=None,
            # y_label=None,
        )

    else:
        if args.mode.lower() == "pairwise":
            if args.layout == "inside":
                num_rows, num_cols = len(algo.dp_mapping), len(algo.dp_mapping)
            elif args.layout == "outside":
                num_rows, num_cols = len(algo.dp_mapping) + 1, len(algo.dp_mapping) + 1
            else:
                raise pyrado.ValueErr(given=args.mode.lower(), eq_constraint="inside or outside")

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 14), tight_layout=False)
            _ = draw_posterior_distr_pairwise(
                axs,
                posterior,
                observations_real,
                algo.dp_mapping,
                condition,
                prior,
                env_real,
                marginal_layout=args.layout,
                grid_res=100,
                normalize_posterior=args.normalize,
                rescale_posterior=args.rescale,
                # x_labels=None,
                # y_labels=None,
            )

        else:
            if args.mode.lower() == "joint":
                num_rows, num_cols = 1, 1
            elif args.mode.lower() in ["separate", "evolution"]:
                num_rows, num_cols = num_rows_cols_from_length(observations_real.shape[0])
            else:
                raise pyrado.ValueErr(given=args.mode, eq_constraint="pairwise, joint, separate, or evolution")

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7), tight_layout=True)
            _ = draw_posterior_distr_2d(
                axs,
                args.mode,
                posterior,
                observations_real,
                algo.dp_mapping,
                idcs_dp,
                prior,
                env_real,
                condition,
                grid_res=200,
                show_prior=False,
                normalize_posterior=args.normalize,
                rescale_posterior=args.rescale,
                add_sep_colorbar=False,
                x_label=None,
                y_label=None,
            )
    if args.save:
        for fmt in ["pdf", "pgf"]:
            os.makedirs(os.path.join(ex_dir, "plots"), exist_ok=True)
            use_rec = "_use_rec" if args.use_rec else ""
            fig.savefig(
                os.path.join(ex_dir, "plots", f"sim_posterior_prob_{args.mode}{use_rec}.{fmt}"),
                dpi=500,
            )
    plt.show()
