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
Plot a real and the simulated rollout segments for evey initial state, i.e. every real-world rollout, using the
`num_ml_samples` most likely domain parameter sets for the segment length of `len_segment` steps.
Use `num_segments = 1` to plot the complete (unsegmented) rollouts.
By default (args.iter = -1), the most recent iteration is evaluated.
"""
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi import NPDR
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.policies.special.time import PlaybackPolicy
from pyrado.sampling.rollout import rollout
from pyrado.spaces import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.checks import check_act_equal
from pyrado.utils.experiments import load_experiment, load_rollouts_from_dir
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--num_segments",
        type=int,
        default=None,
        help="number of segments to split the rollout into (default: None)",
    )
    parser.add_argument(
        "--len_segments",
        type=int,
        default=None,
        help="length of the segments to split the rollout into (default: None)",
    )
    args = parser.parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")
    if not isinstance(args.num_rollouts_per_config, int) or args.num_rollouts_per_config < 1:
        raise pyrado.ValueErr(given=args.num_rollouts_per_config, ge_constraint="1")
    num_ml_samples = args.num_rollouts_per_config

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
    if not isinstance(algo, NPDR):
        raise pyrado.TypeErr(given=algo, expected_type=NPDR)

    # Compute the most likely domain parameters for every target domain observation
    domain_params_ml_all = NPDR.get_ml_posterior_samples(
        algo.dp_mapping,
        posterior,
        observations_real,
        args.num_samples,
        num_ml_samples,
        normalize_posterior=args.normalize,
        sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
    )

    # Load the rollouts
    rollouts_real, _ = load_rollouts_from_dir(ex_dir)
    if args.iter != -1:
        # Only load the selected iteration's initial state
        rollouts_real = rollouts_real[args.iter * algo.num_real_rollouts : (args.iter + 1) * algo.num_real_rollouts]
    num_rollouts_real = len(rollouts_real)
    dim_obs = rollouts_real[0].observations.shape[1]  # same for all rollouts

    if num_rollouts_real > len(domain_params_ml_all):
        print_cbt("Found more real rollouts than sbi observations, truncated the superfluous.", "y")
        rollouts_real = rollouts_real[: len(domain_params_ml_all)]
        num_rollouts_real = len(rollouts_real)
    elif num_rollouts_real < len(domain_params_ml_all):
        print_cbt("Found more sbi observations than real rollouts, truncated the superfluous.", "y")
        domain_params_ml_all = domain_params_ml_all[:num_rollouts_real]

    # Decide on the policy: either use the exact actions or use the same policy which is however observation-dependent
    if args.use_rec:
        policy = PlaybackPolicy(env_sim.spec, [ro.actions for ro in rollouts_real], no_reset=True)

    # Split rollouts into segments
    [ro.numpy() for ro in rollouts_real]
    segments_real_all = []
    for ro in rollouts_real:
        # Split the target domain rollout, see compute_observations()
        if args.num_segments is not None and args.num_segments == 1 and args.len_segments is None:
            segments_real = [ro]
        elif args.num_segments is not None and args.num_segments > 1 and args.len_segments is None:
            segments_real = list(ro.split_ordered_batches(num_batches=args.num_segments + 1))
            segments_real = segments_real[:-1]
        elif args.len_segments is not None and args.num_segments is None:
            segments_real = list(ro.split_ordered_batches(batch_size=args.len_segments))
            if segments_real[-1].length < 2:
                segments_real = segments_real[:-1]
        else:
            raise pyrado.ValueErr(msg="Either batch_size or num_batches must not be None, but not both or none!")

        segments_real_all.append(segments_real)

    # Set the init space of the simulation environment such that we can later set to arbitrary states that could have
    # occurred during the rollout. This is necessary since we are running the evaluation in segments.
    env_sim.init_space = BoxSpace(-pyrado.inf, pyrado.inf, shape=env_sim.init_space.shape)

    # Sample rollouts with the most likely domain parameter sets associated to that observation
    segments_ml_all = []  # all top max likelihood segments for all target domain rollouts
    for idx_r, (segments_real, domain_params_ml) in tqdm(
        enumerate(zip(segments_real_all, domain_params_ml_all)),
        total=len(segments_real_all),
        desc="Sampling",
        file=sys.stdout,
        leave=False,
    ):
        segments_ml = []  # all top max likelihood segments for one target domain rollout
        cnt_step = 0

        # Iterate over target domain segments
        for segment_real in segments_real:
            segments_dp = []  # segments for all domain parameters for the current target domain segment

            # Iterate over domain parameter sets
            for domain_param in domain_params_ml:
                if args.use_rec:
                    # Disabled the policy reset of PlaybackPolicy to do it here manually
                    policy.curr_rec = idx_r
                    policy.curr_step = cnt_step

                # Do the rollout for a segment
                sml = rollout(
                    env_sim,
                    policy,
                    eval=True,
                    reset_kwargs=dict(init_state=segment_real.states[0, :], domain_param=domain_param),
                    max_steps=segment_real.length,
                )
                segments_dp.append(sml)

                assert np.allclose(sml.states[0, :], segment_real.states[0, :])
                if args.use_rec:
                    check_act_equal(segment_real, sml)

            # Increase step counter for next segment, and append all domain parameter segments
            cnt_step += segment_real.length
            segments_ml.append(segments_dp)

        # Append all segments for the current target domain rollout
        segments_ml_all.append(segments_ml)

    assert len(segments_ml_all) == len(segments_real_all)

    # Sample rollouts using the nominal domain parameters
    if args.use_rec:
        policy.reset_curr_rec()
    env_sim.domain_param = env_sim.get_nominal_domain_param()
    segments_nom = []
    for idx_r, segments_real in enumerate(segments_real_all):
        segment_nom = []
        cnt_step = 0
        for segment_real in segments_real:
            if args.use_rec:
                policy.curr_rec = idx_r  # counter the policy.reset() in rollout()
                policy.curr_step = cnt_step
                cnt_step += segment_real.length

            # Do the rollout for a segment
            sn = rollout(
                env_sim,
                policy,
                eval=True,
                reset_kwargs=dict(init_state=segment_real.states[0]),
                max_steps=segment_real.length,
            )
            segment_nom.append(sn)
            if args.use_rec:
                check_act_equal(segment_real, sn)

        # Append individual segments
        segments_nom.append(segment_nom)

    assert len(segments_nom) == len(segments_ml_all)

    # Plot the different init states in separate figures and the different observations along the columns
    colors = plt.get_cmap("Reds")(np.linspace(0.5, 1.0, num_ml_samples))
    for idx_r in range(len(segments_real_all)):
        fig, axs = plt.subplots(nrows=dim_obs, figsize=(16, 9), tight_layout=True, sharex="col")

        for idx_o in range(dim_obs):
            # Plot the real segments
            cnt_step = [0]
            for segment_real in segments_real_all[idx_r]:
                axs[idx_o].plot(
                    np.arange(cnt_step[-1], cnt_step[-1] + segment_real.length),
                    segment_real.get_data_values("observations", True)[:, idx_o],
                    c="black",
                    label="real" if cnt_step[-1] == 0 else "",  # only print once
                )
                cnt_step.append(cnt_step[-1] + segment_real.length)

            # Plot the maximum likely simulated segments
            for idx_s, sml in enumerate(segments_ml_all[idx_r]):
                for idx_dp, smdp in enumerate(sml):
                    axs[idx_o].plot(
                        np.arange(cnt_step[idx_s], cnt_step[idx_s] + smdp.length),
                        smdp.get_data_values("observations", True)[:, idx_o],
                        c=colors[idx_dp],
                        ls="--",
                        label=f"sim ml {idx_dp}" if cnt_step[idx_s] == 0 else "",  # only print once for each dp set
                    )

            # Plot the nominal simulation's segments
            for idx_s, sn in enumerate(segments_nom[idx_r]):
                axs[idx_o].plot(
                    np.arange(cnt_step[idx_s], cnt_step[idx_s] + sn.length),
                    sn.get_data_values("observations", True)[:, idx_o],
                    c="steelblue",
                    ls="-.",
                    label="sim nom" if cnt_step[idx_s] == 0 else "",  # only print once
                )

        # Set window title and the legend, placing the latter above the plot expanding and expanding it fully
        fig.canvas.set_window_title(f"Real and Simulated Observations (rollout {idx_r+1} of {num_rollouts_real})")
        lg = axs[0].legend(
            ncol=2 + num_ml_samples,
            bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
            loc="lower left",
            mode="expand",
            borderaxespad=0.0,
        )

        if args.save:
            for fmt in ["pdf", "pgf"]:
                os.makedirs(os.path.join(ex_dir, "plots"), exist_ok=True)
                use_rec = "_use_rec" if args.use_rec else ""
                fig.savefig(
                    os.path.join(ex_dir, "plots", f"sim_real_rollouts_{idx_r}{use_rec}.{fmt}"),
                    bbox_extra_artists=(lg,),
                    dpi=500,
                )

    plt.show()
