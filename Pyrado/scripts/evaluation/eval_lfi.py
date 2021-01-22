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
Script to evaluate a posterior obtained using the sbi package
"""
import os
import torch as to
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi import LFI
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.distribution import draw_posterior_distr
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")
    if args.mode not in ["joint", "separate"]:
        raise pyrado.ValueErr(given=args.mode, given_name="plotting mode", eq_constraint="joint or separate")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the environments, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)
    env_real = pyrado.load(None, "env_real", "pkl", ex_dir)
    prior = kwout["prior"]
    posterior = kwout["posterior"]
    observations_real = kwout["observations_real"]

    # Load the algorithm and the required data
    algo = Algorithm.load_snapshot(ex_dir)
    if not isinstance(algo, LFI):
        raise pyrado.TypeErr(given=algo, expected_type=LFI)

    # Select the domain parameters to plot
    if len(algo.dp_mapping) > 2:
        usr_inp = input(
            f"Found the domain parameter mapping {algo.dp_mapping}. Select 2 domain parameter by index "
            f"to be plotted (format: separated by a whitespace):\n"
        )
        dp_idcs = tuple(map(int, usr_inp.split()))
    else:
        dp_idcs = (0, 1)

    # Load a specific real-world observation (off, i.e. -1, by default)
    if args.iter != -1:
        # Crawl through the experiment's directory
        for root, dirs, files in os.walk(ex_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            found_observations = [o for o in files if o.startswith("iter_") and o.endswith("_observations_real.pt")]
        load_iter = len(found_observations) - 1
        observations_real = pyrado.load(None, f"iter_{load_iter}_observations_real", "pt", ex_dir)

    # Evaluate the posterior
    domain_params, log_prob, _ = LFI.eval_posterior(
        posterior, observations_real, args.num_samples, algo.sbi_simulator, simulate_observations=False
    )

    # Set the condition if necessary
    if len(algo.dp_mapping) > 2:
        condition = to.mean(domain_params, dim=[0, 1]) # to.median(to.median(domain_params, dim=0)[0], dim=0)[0]
    else:
        condition = None


    # Plot the posterior distribution, the true parameters / their distribution
    if args.mode.lower() == "joint":
        num_rows, num_cols = 1, 1
    else:
        num_rows, num_cols = num_rows_cols_from_length(observations_real.shape[0])
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(14, 7), tight_layout=True)
    _ = draw_posterior_distr(
        axs,
        args.mode.lower(),
        posterior,
        observations_real,
        algo.dp_mapping,
        env_real,
        prior,
        dp_idcs,
        condition,
        show_prior=False,
        # grid_bounds=to.tensor([[0.1, 0.5], [1, 3.5]])
    )

    plt.show()
