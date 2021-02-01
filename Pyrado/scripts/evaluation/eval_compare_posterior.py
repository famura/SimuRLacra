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
Comparison is carried out via Maximum-Mean Discrepancy.
This script will only be successful if the simulator is tractable i.e. its probability can be evaluated.
"""
import os
import os.path as osp
import torch as to
import pandas as pd
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi import LFI
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.distribution import draw_posterior_distr
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment

from pyrado.sampling.posterior_sampler import posterior_sampler, rejection_sampler

from geomloss.samples_loss import SamplesLoss

if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument("--load_data", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the environments, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)

    env_real = pyrado.load(None, "env_real", "pkl", ex_dir)
    # check if environment is wrapped into DomainRandWrapperBuffer
    env_real = env_real.wrapped_env if isinstance(env_real, DomainRandWrapper) else env_real

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

    reference_dir = osp.join(ex_dir, f"../../../../perma/environment/{env_real.name}/reference")
    if not osp.isdir(reference_dir):
        raise pyrado.PathErr(given=reference_dir)

    num_observation = 10
    reference_samples = [
        pyrado.load(None, "reference_posterior_samples", "pt", osp.join(reference_dir, f"num_observation_{n}"))
        for n in range(1, num_observation + 1)
    ]
    reference_samples = to.stack(reference_samples)[:, : args.num_samples, :]
    observation = [
        pyrado.load(None, "observation", "pt", osp.join(reference_dir, f"num_observation_{n}"))
        for n in range(1, num_observation + 1)
    ]
    observation = to.stack(observation)
    true_params = [
        pyrado.load(None, "true_parameters", "pt", osp.join(reference_dir, f"num_observation_{n}"))
        for n in range(1, num_observation + 1)
    ]
    true_params = to.stack(true_params)

    def eval(curr_iter=0, num_samples=None, mmd_mean=None, mmd_std=None, posterior_samples=None):
        num_samples = [] if num_samples is None else num_samples
        mmd_mean = [] if mmd_mean is None else mmd_mean
        mmd_std = [] if mmd_std is None else mmd_std
        posterior_samples = [] if posterior_samples is None else posterior_samples

        for iter in range(curr_iter, load_iter):
            # Compute and print the argmax
            print(f"Current Iter:\t{iter}")
            # generate postserior samples
            param_samples, _, _ = LFI.eval_posterior(
                posterior, observation, args.num_samples, algo.sbi_simulator, simulate_observations=False, calculate_log_probs=False
            )

            # calculate mmd-loss
            with to.no_grad():
                mmd_loss = SamplesLoss(loss="energy")
                loss = mmd_loss(reference_samples, param_samples)

            # save progress
            mmd_mean.append(loss.mean(dim=0).item())
            mmd_std.append(loss.std(dim=0).item())
            num_samples.append(iter * num_sim_per_real_rollout * num_real_rollouts)
            posterior_samples.append(param_samples)

            pyrado.save(num_samples, "num_samples", "pt", ex_dir)
            pyrado.save(mmd_mean, "mmd_mean", "pt", ex_dir)
            pyrado.save(mmd_std, "mmd_std", "pt", ex_dir)
            pyrado.save(posterior_samples, "posterior_samples", "pt", ex_dir)

    mmd_mean, mmd_std, num_samples, posterior_samples = [], [], [], []
    check_files = ["num_samples", "mmd_mean", "mmd_std", "posterior_samples"]
    if args.load_data:
        pass
    elif args.resume:
        check_files = ["num_samples", "mmd_mean", "mmd_std", "posterior_samples"]
        num_samples, mmd_mean, mmd_std, posterior_samples = [
            pyrado.load(None, cf, "pt", ex_dir) for cf in check_files
        ]
        curr_iter = int(num_samples[-1] / num_sim_per_real_rollout / num_real_rollouts) + 1
        eval(
            curr_iter=curr_iter,
            num_samples=num_samples,
            mmd_mean=mmd_mean,
            mmd_std=mmd_std,
            posterior_samples=posterior_samples,
        )
    else:
        eval()
