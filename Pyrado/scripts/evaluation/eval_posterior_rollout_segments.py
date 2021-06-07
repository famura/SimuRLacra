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
By default (args.iter = -1), the all iterations are evaluated.
"""
import os.path as osp
from typing import Optional

import dtw
import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.environment_wrappers.domain_randomization import remove_all_dr_wrappers
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.rollout_based import plot_rollouts_segment_wise
from pyrado.policies.feed_forward.playback import PlaybackPolicy
from pyrado.sampling.parallel_evaluation import eval_domain_params_with_segmentwise_reset
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.sampling.step_sequence import StepSequence, check_act_equal
from pyrado.spaces.box import InfBoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import merge_dicts, repeat_interleave
from pyrado.utils.experiments import load_experiment, load_rollouts_from_dir
from pyrado.utils.input_output import print_cbt
from pyrado.utils.math import rmse


def compute_traj_distance_metrics(
    states_real: np.ndarray,
    states_ml: np.ndarray,
    states_nom: np.ndarray,
    num_rollouts_real: int,
    normalize: bool = True,
    dtw_config: Optional[dict] = None,
    save: bool = True,
):
    """
    Compute the DTW distance and the RMSE for 2 trajectories w.r.t. a ground truth trajectory, and store it in a table.

    :param states_real: numpy array of states from the real system of shape [num_rollouts, len_time_series, dim_state]
    :param states_ml: numpy array of states from the most likely system [num_rollouts, len_time_series, dim_state]
    :param states_nom: numpy array of states from the nominal system [num_rollouts, len_time_series, dim_state]
    :param num_rollouts_real: number of rollouts
    :param normalize: it `True`, normalize all trajectories by the max. abs. values of the ground truth trajectory for
                      each dimension before computing the metrics
    :param dtw_config: dictionary with options for the `dtw.dtw()` command, e.g.
                       `dict(step_pattern=dtw.rabinerJuangStepPattern(6, "c"))`
    :param save: it `True`, save table as tex-file
    """
    # Configure the metric computations
    default_dtw_config = dict(open_end=True, step_pattern="symmetric2", distance_only=True)
    dtw_config = merge_dicts([default_dtw_config, dtw_config or dict()])

    # Iterate over all rollouts and compute the performance metrics
    table = []
    dtw_dist_ml_avg, dtw_dist_nom_avg, rmse_ml_avg, rmse_nom_avg = 0, 0, 0, 0
    for idx_r in range(num_rollouts_real):
        if normalize:
            # Normalize all trajectories by the max. abs. values of the ground truth trajectory for each dimension
            max_abs_state = np.max(np.abs(states_real[idx_r]), axis=0)
            states_real[idx_r] /= max_abs_state
            states_ml[idx_r] /= max_abs_state
            states_nom[idx_r] /= max_abs_state

        # DTW
        dtw_dist_ml = dtw.dtw(states_real[idx_r], states_ml[idx_r], **dtw_config).distance
        dtw_dist_nom = dtw.dtw(states_real[idx_r], states_nom[idx_r], **dtw_config).distance
        dtw_dist_ml_avg += dtw_dist_ml / num_rollouts_real
        dtw_dist_nom_avg += dtw_dist_nom / num_rollouts_real

        # RMSE averaged over the states
        rmse_ml = np.mean(rmse(states_real[idx_r], states_ml[idx_r], dim=0))
        rmse_nom = np.mean(rmse(states_real[idx_r], states_nom[idx_r], dim=0))
        rmse_ml_avg += rmse_ml / num_rollouts_real
        rmse_nom_avg += rmse_nom / num_rollouts_real

        table.append([idx_r, dtw_dist_ml, dtw_dist_nom, rmse_ml, rmse_nom])

    # Add last row separately
    table.append(["average", dtw_dist_ml_avg, dtw_dist_nom_avg, rmse_ml_avg, rmse_nom_avg])

    # Print the tabulated data
    headers = ("rollout", "DTW dist. ml", "DTW dist. nom", "mean RMSE ml", "mean RMSE nom")
    print(tabulate(table, headers))

    if save:
        # Save the table for LaTeX
        table_latex_str = tabulate(table, headers, tablefmt="latex")
        str_iter = f"_iter_{args.iter}"
        str_round = f"_round_{args.round}"
        use_rec_str = "_use_rec" if args.use_rec else ""
        with open(osp.join(ex_dir, f"distance_metrics{str_iter}{str_round}{use_rec_str}.tex"), "w") as tab_file:
            print(table_latex_str, file=tab_file)


if __name__ == "__main__":
    parser = get_argparser()
    parser.add_argument(
        "--data_type",
        type=str,
        default="states",
        help="select data type for plotting, e.g. 'observations' or 'states' (default: 'states')",
    )

    parser.add_argument(
        "--save_format",
        nargs="+",
        type=str,
        default=["pdf", "pgf", "png"],
        help="select file format for plot saving, without commas (e.g., 'pdf png')",
    )

    parser.add_argument(
        "--console",
        action="store_true",
        default=False,
        help="set flag to not run plt.show. Make sure that the --save flag is set",
    )

    # Parse command line arguments
    args = parser.parse_args()
    if not isinstance(args.num_samples, int) or args.num_samples < 1:
        raise pyrado.ValueErr(given=args.num_samples, ge_constraint="1")
    if not isinstance(args.num_rollouts_per_config, int) or args.num_rollouts_per_config < 1:
        raise pyrado.ValueErr(given=args.num_rollouts_per_config, ge_constraint="1")
    num_ml_samples = args.num_rollouts_per_config
    if not args.mode.lower() in ["samples", "confidence"]:
        raise pyrado.ValueErr(given=args, eq_constraint="samples or confidence")
    if args.cut_rollout is not None:
        if len(args.cut_rollout) != 2:
            raise pyrado.ValueErr(given=args.cut_rollout, eq_constraint="tuple of integers")
        else:
            args.cut_rollout = tuple(args.cut_rollout)

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the environments, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)
    env_sim = remove_all_dr_wrappers(env_sim)  # randomize manually later
    env_real = pyrado.load("env_real.pkl", ex_dir)
    prior = kwout["prior"]
    posterior = kwout["posterior"]
    data_real = kwout["data_real"]

    # Load the algorithm and the required data
    algo = Algorithm.load_snapshot(ex_dir)
    if not isinstance(algo, (NPDR, BayesSim)):
        raise pyrado.TypeErr(given=algo, expected_type=(NPDR, BayesSim))

    # Set seed if desired
    pyrado.set_seed(args.seed)

    # Load the rollouts
    rollouts_real, _ = load_rollouts_from_dir(ex_dir)
    if args.iter != -1:
        # Only load the selected iteration's rollouts
        rollouts_real = rollouts_real[args.iter * algo.num_real_rollouts : (args.iter + 1) * algo.num_real_rollouts]
    num_rollouts_real = len(rollouts_real)
    [ro.numpy() for ro in rollouts_real]
    dim_state = rollouts_real[0].states.shape[1]  # same for all rollouts

    # Decide on the policy: either use the exact actions or use the same policy which is however observation-dependent
    if args.use_rec:
        policy = PlaybackPolicy(env_sim.spec, [ro.actions for ro in rollouts_real], no_reset=True)

    # Compute the most likely domain parameters for every target domain observation
    domain_params_ml_all, _ = SBIBase.get_ml_posterior_samples(
        algo.dp_mapping,
        posterior,
        data_real,
        num_eval_samples=args.num_samples,
        num_ml_samples=num_ml_samples,
        normalize_posterior=args.normalize,
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=args.use_mcmc),
    )

    # Repeat the domain parameters to zip them later with the real rollouts, such that they all belong to the same iter
    num_iter = len(domain_params_ml_all)
    num_rep = num_rollouts_real // num_iter
    domain_params_ml_all = repeat_interleave(domain_params_ml_all, num_rep)
    assert len(domain_params_ml_all) == num_rollouts_real

    # Split rollouts into segments
    segments_real_all = []
    for ro in rollouts_real:
        # Split the target domain rollout, see SimRolloutSamplerForSBI.__call__()
        if args.num_segments is not None and args.len_segments is None:
            segments_real = list(ro.split_ordered_batches(num_batches=args.num_segments))
        elif args.len_segments is not None and args.num_segments is None:
            segments_real = list(ro.split_ordered_batches(batch_size=args.len_segments))
        else:
            raise pyrado.ValueErr(msg="Either num_segments or len_segments must not be None, but not both or none!")

        segments_real_all.append(segments_real)

    # Set the init space of the simulation environment such that we can later set to arbitrary states that could have
    # occurred during the rollout. This is necessary since we are running the evaluation in segments.
    env_sim.init_space = InfBoxSpace(shape=env_sim.init_space.shape)

    # New solution with parallelization of the rollouts. Has been double-checked.
    if True:
        # Create a new sampler pool for every policy to synchronize the random seeds i.e. init states
        pool = SamplerPool(args.num_workers)

        # Seed the sampler
        if args.seed is not None:
            pool.set_seed(args.seed)
            print_cbt(f"Set the random number generators' seed to {args.seed}.", "w")
        else:
            print_cbt("No seed was set", "y")

        # Append all segments for the current target domain rollout
        segments_ml_all = eval_domain_params_with_segmentwise_reset(
            pool, env_sim, policy, segments_real_all, domain_params_ml_all, algo.stop_on_done, args.use_rec
        )

    # Old solution without parallelization of the rollouts. Only use this when deeply mistrusting the one above.
    else:
        import sys

        from tqdm import tqdm

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
                    sdp = rollout(
                        env_sim,
                        policy,
                        eval=True,
                        reset_kwargs=dict(init_state=segment_real.states[0, :], domain_param=domain_param),
                        max_steps=segment_real.length,
                        stop_on_done=algo.stop_on_done,
                    )
                    segments_dp.append(sdp)

                    assert np.allclose(sdp.states[0, :], segment_real.states[0, :])
                    if args.use_rec:
                        check_act_equal(segment_real, sdp)

                    # Pad if necessary
                    StepSequence.pad(sdp, segment_real.length)

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
                stop_on_done=algo.stop_on_done,
            )
            segment_nom.append(sn)
            if args.use_rec:
                check_act_equal(segment_real, sn)

            # Pad if necessary
            StepSequence.pad(sn, segment_real.length)

        # Append individual segments
        segments_nom.append(segment_nom)
    assert len(segments_nom) == len(segments_ml_all)

    # Get the states for computing the performance metrics
    states_real = np.stack([ro.get_data_values("states", truncate_last=True) for ro in rollouts_real], axis=0)
    states_nom = np.stack([StepSequence.concat(segs_nom).states for segs_nom in segments_nom], axis=0)
    states_ml = np.stack(
        [
            StepSequence.concat([s[0] for s in [segs_ml for segs_ml in segments_ml]]).states  # 0 ist the most likely
            for segments_ml in segments_ml_all
        ],
        axis=0,
    )
    assert states_real.shape == states_nom.shape == states_ml.shape
    assert states_real.shape[0] == num_rollouts_real

    # Compute the DTW and RMSE distance and store it in a table
    table = compute_traj_distance_metrics(states_real, states_ml, states_nom, num_rollouts_real)

    # Plot
    plot_rollouts_segment_wise(
        args.mode.lower(),
        segments_real_all,
        segments_ml_all,
        segments_nom,
        use_rec_str=args.use_rec,
        idx_iter=args.iter,
        idx_round=args.round,
        show_act=False,
        save_dir=ex_dir if args.save else None,
        x_limits=args.cut_rollout,
        data_field=args.data_type,
        file_format=args.save_format,
    )

    if not args.console:
        plt.show()
