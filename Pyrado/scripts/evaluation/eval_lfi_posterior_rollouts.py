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
Plot a rollout for evey initial state using the most likely domain parameter set.
"""
import os
import torch as to
from matplotlib import pyplot as plt

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi import LFI
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment, load_rollouts_from_dir
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the environments, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)
    env_sim = remove_all_dr_wrappers(env_sim)  # randomize manually later
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
    domain_params_ml = []
    for i in range(domain_params.shape[0]):
        dp_val = domain_params[i, to.argmax(log_probs[i, :]), :].numpy()
        domain_params_ml.append(dict(zip(algo.dp_mapping.values(), dp_val)))

    # Load the rollouts
    rollouts_real = load_rollouts_from_dir(ex_dir)
    if not rollouts_real:
        raise pyrado.ValueErr(msg="No rollouts have been found!")

    # Extract init states
    [ro.numpy() for ro in rollouts_real]
    init_states_real = [ro.rollout_info["init_state"] for ro in rollouts_real]
    num_rollouts_real = len(init_states_real)
    if num_rollouts_real > len(domain_params_ml):
        print_cbt("Found more init states than sbi observations, truncated the superfluous.", "y")
        init_states_real = init_states_real[: len(domain_params_ml), :]

    # Sample rollouts with the most likely domain parameter set associated to that observation
    rollouts_sim = []
    for init_state, domain_param in zip(init_states_real, domain_params_ml):
        rollouts_sim.append(
            rollout(env_sim, policy, eval=True, reset_kwargs=dict(init_state=init_state, domain_param=domain_param))
        )
    assert len(rollouts_sim) == len(rollouts_real)

    # Sample rollouts using the nominal domain parameters
    env_sim.domain_param = env_sim.get_nominal_domain_param()
    rollouts_nom_sim = []
    for init_state in init_states_real:
        rollouts_nom_sim.append(
            rollout(env_sim, policy, eval=True, reset_kwargs=dict(init_state=init_state))
        )
    assert len(rollouts_nom_sim) == len(rollouts_sim)

    # Plot the different init states in separate figures and the different observations along the columns
    dim_obs = rollouts_real[0].observations.shape[1]  # same for all rollouts
    for idx_r in range(num_rollouts_real):
        fig, axs = plt.subplots(nrows=dim_obs, figsize=(16, 9), tight_layout=True, sharex="col")
        for idx_o in range(dim_obs):
            # Plot the real rollouts
            axs[idx_o].plot(rollouts_real[idx_r].observations[:, idx_o], label="real")
            # Plot the maximum likely simulated rollouts for every real init state
            axs[idx_o].plot(rollouts_sim[idx_r].observations[:, idx_o], ls="--", label="sim ml")
            # Plot the nominal simulation's rollouts for every real init state
            axs[idx_o].plot(rollouts_nom_sim[idx_r].observations[:, idx_o], ls="-.", label="sim nom", c="grey")

        # Set legend and window title
        axs[0].legend(ncol=3)
        fig.canvas.set_window_title(f"Real and Simulated Observations (rollout {idx_r+1} of {num_rollouts_real})")

        if args.save_figures:
            for fmt in ["pdf", "pgf"]:
                fig.savefig(os.path.join(ex_dir, f"rollout_comp_lfi_sim_real_{idx_r}.{fmt}"), dpi=500)

    plt.show()
