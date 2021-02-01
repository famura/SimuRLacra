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
import numpy as np
import os
import torch as to
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi import LFI
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper, remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment, load_rollouts_from_dir


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

    # Load a specific real-world observation (off, i.e. -1, by default)
    if args.iter != -1:
        # Crawl through the experiment's directory
        for root, dirs, files in os.walk(ex_dir):
            dirs.clear()  # prevents walk() from going into subdirectories
            found_observations = [o for o in files if o.startswith("iter_") and o.endswith("_observations_real.pt")]
        load_iter = len(found_observations) - 1
        observations_real = pyrado.load(None, f"iter_{load_iter}_observations_real", "pt", ex_dir)

    domain_params, log_probs, _ = LFI.eval_posterior(
        posterior,
        observations_real,
        args.num_samples,
        algo.sbi_simulator,
        normalize_posterior=False,
        simulate_observations=False,
    )

    # Extract the most likely domain parameter sets for every real-world
    dp_ml = []
    for i in range(domain_params.shape[0]):
        dp_ml.append(domain_params[i, to.argmax(log_probs[i, :]), :])
    dp_ml = to.stack(dp_ml)

    # Load the rollouts
    rollouts_real = load_rollouts_from_dir(ex_dir)
    if not rollouts_real:
        raise pyrado.ValueErr(msg="No rollouts have been found!")

    # Extract init states
    [ro.numpy() for ro in rollouts_real]
    init_states_real = [ro.rollout_info["init_state"] for ro in rollouts_real]

    env_sim = remove_all_dr_wrappers(env_sim)
    env_sim.domain_param = dict(zip(algo.dp_mapping.values(), dp_ml.numpy()))

    # Sample rollouts
    sampler = ParallelRolloutSampler(env_sim, policy, num_workers=1, min_rollouts=1)
    rollouts_sim = sampler.sample(init_states=init_states_real)

    assert all(
        [
            np.allclose(r.rollout_info["init_state"], s.rollout_info["init_state"])
            for r, s in zip(rollouts_real, rollouts_sim)
        ]
    )

    obs_real = [ro.observations for ro in rollouts_real]
    obs_sim = [ro.observations for ro in rollouts_sim]

    # Plot a rollout for evey inital state using the most likely domain parameter set
    fig, axs = plt.subplots(num_rows=1, num_cols=1, figsize=(14, 7), tight_layout=True)

    plt.show()
