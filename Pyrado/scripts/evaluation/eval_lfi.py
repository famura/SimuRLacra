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

from sbi import utils as utils

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi2 import LFI
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import ask_for_experiment
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the environment, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)
    prior = kwout["prior"]
    posterior = kwout["posterior"]
    observations_real = kwout["observations_real"]
    # Load the algorithm and the required data
    algo = Algorithm.load_snapshot(ex_dir)
    if not isinstance(algo, LFI):
        raise pyrado.TypeErr(given=algo, expected_type=LFI)
    algo.setup_sbi()

    # Reconstruct ground truth domain parameters if they exist
    env_real = pyrado.load(None, "env_real", "pkl", ex_dir)
    if typed_env(env_real, DomainRandWrapperBuffer):
        domain_param_gt = [to.stack(list(d.values())) for d in env_real.randomizer.get_params(-1, "list", "torch")]
    elif isinstance(env_real, SimEnv):
        domain_param_gt = to.tensor([env_real.domain_param[v] for v in algo.dp_mapping.values()])
    else:
        domain_param_gt = None

    # Load a specific real-world observation (by default the latest)
    if args.iter == -1:
        # Crawl through the experiment's directory
        for root, dirs, files in os.walk(ex_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            found_observations = [o for o in files if o.startswith("iter_") and o.endswith("_observations_real.pt")]
        load_iter = len(found_observations) - 1
    else:
        load_iter = args.iter
    observations_real_sel = pyrado.load(None, f"iter_{load_iter}_observations_real", "pt", ex_dir)

    # Compute and print the argmax
    domain_params, log_prob, _ = LFI.eval_posterior(
        posterior, observations_real_sel, args.num_samples, algo.sbi_simulator, simulate_observations=False
    )

    # TODO whatever you wanna do here

    fig, axes = utils.pairplot(
        domain_params[args.iter, :, :],
        limits=[[23, 43], [0, 0.8]],
        fig_size=(8, 8),
        points=domain_param_gt[args.iter],
        points_offdiag={"markersize": 6},
        points_colors="r",
    )
    axes[0,1].axhline(y=prior.support.lower_bound[0], c="w", ls="--")
    axes[0,1].axhline(y=prior.support.upper_bound[0], c="w", ls="--")
    axes[0,1].axvline(x=prior.support.lower_bound[1], c="w", ls="--")
    axes[0,1].axvline(x=prior.support.upper_bound[1], c="w", ls="--")

    # -------------- WIP

    resolution = 100
    dp_1 = to.linspace(23, 43, resolution)
    dp_2 = to.linspace(0, 0.8, resolution)
    x_grid, y_grid = to.meshgrid([dp_1, dp_2])
    x_grid, y_grid = x_grid.t(), y_grid.t()  # transpose not necessary but makes identical mesh as np.meshgrid
    grid = to.cat((x_grid.reshape(-1, 1), y_grid.reshape(-1, 1)), dim=1)
    if not grid.shape == (resolution**2, 2):
        raise pyrado.ShapeErr(given=grid, expected_match=(resolution**2, 2))

    log_probs = posterior.log_prob(grid, x=observations_real_sel[0])  # TODO sum over multiple observations
    probs = to.exp(log_probs - log_probs.max())  # scale the probabilities to [0, 1]
    probs = probs.reshape(resolution, resolution).numpy()

    # import numpy as np
    # dp_1 = np.linspace(23, 43, 100, endpoint=True)
    # dp_2 = np.linspace(0, 0.8, 100, endpoint=True)
    # x_grid2, y_grid2 = np.meshgrid(dp_1, dp_2)

    plt.show()
