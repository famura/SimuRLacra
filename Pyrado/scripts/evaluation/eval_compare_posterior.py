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
Comparison is carried out via Maximum-Mean Discrpency.
This script will only be succesful if the simulator is tractable i.e. its probability can be evaluated.
"""
import os
import torch as to
import pandas as pd
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi2 import LFI
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.distribution import draw_posterior_distr
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment

from pyrado.environments.one_step.multivariate_gaussian import ToyExample
from pyrado.sampling.posterior_sampler import posterior_sampler

from geomloss.samples_loss import SamplesLoss


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

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

    # algo hparams
    params_names = list(algo.dp_mapping.values())
    num_sim_per_real_rollout = algo.num_sim_per_real_rollout
    num_real_rollouts = algo.num_real_rollouts

    # Load a specific real-world observation (by default the latest)
    if args.iter == -1:
        # Crawl through the experiment's directory
        for root, dirs, files in os.walk(ex_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            found_observations = [o for o in files if o.startswith("iter_") and o.endswith("_observations_real.pt")]
        load_iter = len(found_observations) - 1
    else:
        load_iter = args.iter

    # go through each iteration and calculate the mmd
    num_samples = []
    mmd_mean = []
    mmd_std = []
    true_samples = None
    posterior_samples = None
    observations_real_sel = None
    for iter in range(load_iter):
        print(f"current iter: {iter}")
        observations_real_sel = pyrado.load(None, f"iter_{iter}_observations_real", "pt", ex_dir)
        posterior = pyrado.load(None, f"iter_{iter}_posterior", "pt", ex_dir)

        # sample true posterior samples
        true_samples = posterior_sampler(env_real, prior, observations_real_sel, params_names, num_samples=args.num_samples)


        # Compute and print the argmax
        posterior_samples, _, _ = LFI.eval_posterior(
            posterior, observations_real_sel, args.num_samples, algo.sbi_simulator, simulate_observations=False
        )
        with to.no_grad():
            mmd_loss = SamplesLoss(loss="energy")
            loss = mmd_loss(true_samples, posterior_samples)
        mmd_mean.append(loss.mean().item())
        mmd_std.append(loss.std().item())
        num_samples.append(iter * num_sim_per_real_rollout * num_real_rollouts)

    num_samples = to.tensor(num_samples)
    mmd_mean = to.tensor(mmd_mean)
    mmd_std = to.tensor(mmd_std)

    pyrado.save(num_samples, "num_samples", "pt", ex_dir)
    pyrado.save(mmd_mean, "mmd_mean", "pt", ex_dir)
    pyrado.save(mmd_std, "mmd_std", "pt", ex_dir)

    # plot trainings progress with mmd
    plt.figure()
    plt.plot(num_samples, mmd_mean, mmd_std)
    plt.fill_between(num_samples, mmd_mean - mmd_std, mmd_mean + mmd_std,
                     color='gray', alpha=0.2)
    plt.show()

    # plot scatter of the true posterior
    plt.figure()
    plt.scatter(true_samples[0, :, 0], true_samples[0, :, 1], color="black")
    plt.scatter(posterior_samples[0, :, 0], posterior_samples[0, :, 1], color="blue")
    plt.show()

    # TODO: use draw_posterior_distribution but it requires marginalizing over states which are not depicted.
    #  For now there is only a scatter-plot of the domain-parameters




