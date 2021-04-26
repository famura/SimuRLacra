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

import argparse


def get_argparser() -> argparse.ArgumentParser:
    """Return Pyrado's default argument parser."""

    parser = argparse.ArgumentParser(
        description="Pyrado's default argument parser",
    )
    parser.add_argument(
        "--animation",
        dest="animation",
        action="store_true",
        help="show a rendered animation (default: True)",
    )
    parser.add_argument(
        "--no_animation",
        dest="animation",
        action="store_false",
    )
    parser.set_defaults(animation=True)
    parser.add_argument(
        "--dt",
        type=float,
        help="environments time step size in seconds (no default)",
    )
    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        nargs="?",
        help="path to the (experiment) directory to load from",
    )
    parser.add_argument(
        "-e",
        "--env_name",
        type=str,
        nargs="?",
        help="name of the environment to use (e.g. 'qbb' or 'qcp-st')",
    )
    parser.add_argument(
        "--idcs",
        nargs="+",
        type=int,
        default=[0, 1],
        help="list of indices without commas casted to integer (default: [0, 1])",
    )
    parser.add_argument(
        "--init_state",
        nargs="+",
        type=float,
        default=None,
        help="list of init state values without commas, e.g. '1.2 3 0.9' (default: None)",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=-1,
        help="iteration to select for evaluation (default: -1)",
    )
    parser.add_argument(
        "--layout",
        default="outside",
        help="plotting layout for plotting, e.g. inside or outside for plotting a posterior (default: outside)",
    )
    parser.add_argument(
        "--load_all",
        action="store_true",
        default=False,
        help="load all quantities e.g. policies (default: False)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=float("inf"),
        help="maximum number of time steps to execute the environment (default: infinite)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        nargs="?",
        help="general argument to specify different modes of various scripts (e.g. '2D')",
    )
    parser.add_argument(
        "--use_mcmc",
        action="store_true",
        default=False,
        help="Use Markov Chain Monte-Carlo for sampling from the posterior (default: False)",
    )
    parser.add_argument(
        "--no_dr",
        action="store_true",
        default=False,
        help="remove all domain randomization wrappers (default: False)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="normalize sth (depending on the script), e.g. the log-probabilities of the posterior (default: False)",
    )
    parser.add_argument(
        "--num_rollouts_per_config",
        type=int,
        help="number of rollouts per environment configuration, e.g. domain parameter sets or initial states",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of environments to sample from in parallel (default: 4)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="number of runs for the overall experiment (default: 1)",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        help="number of samples",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="verbose",
        action="store_false",
        help="display minimal information, the opposite of verbose (default: True)",
    )
    parser.add_argument(
        "--random_init_state",
        action="store_true",
        default=False,
        help="use a random initial state (default: False)",
    )
    parser.add_argument(
        "--relentless",
        action="store_true",
        default=False,
        help="don't stop (e.g. continue simulating after done flag was raised)",
    )
    parser.add_argument(
        "--render",
        dest="render",
        action="store_true",
        help="show a rendered animation (default: True)",
    )
    parser.add_argument(
        "--no_render",
        dest="render",
        action="store_false",
    )
    parser.set_defaults(render=False)
    parser.add_argument(
        "--rescale",
        action="store_true",
        default=False,
        help="rescale sth (depending on the script), e.g. the posterior probabilities to be in ]0, 1] (default: False)",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=-1,
        help="round index, e.g. for the NPDR algorithm (default: -1)",
    )
    parser.add_argument(
        "--policy_name",
        type=str,
        nargs="?",
        default="policy",
        help="(partial) name of the policy to load, e.g. 'argmax_policy', or 'iter_0_policy' " "(default: policy)",
    )
    parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        default=False,
        help="save all generated figures (default: False)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for the random number generators (default: None for no seeding)",
    )
    parser.add_argument(
        "--use_rec",
        action="store_true",
        default=False,
        help="use pre-recorded data, for example a sequence of actions (default: False)",
    )
    parser.add_argument(
        "--use_tex",
        action="store_true",
        default=False,
        help="use LaTeX fonts for plotting text with matplotlib (default: False)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="display additional information (default: False)",
    )
    parser.set_defaults(verbose=False)
    parser.add_argument(
        "--vfcn_name",
        type=str,
        nargs="?",
        default="vfcn",
        help="(partial) name of the value function to load, e.g. 'argmax_vfcn', or 'iter_0_vfcn' (default: vfcn)",
    )
    parser.add_argument(
        "--warmstart",
        dest="warmstart",
        action="store_true",
        help="start a procedure with initialized parameters, e.g. for the policy",
    )
    parser.add_argument(
        "--from_scratch",
        dest="warmstart",
        action="store_false",
        help="the opposite of 'warmstart'",
    )
    parser.add_argument(
        "-p",
        "--show_hparams",
        action="append",
        help="hyperparameters to show in the ask-for-experiment dialog; use this parameter multiple times to show multiple hyperparameters; e.g. to differentiate between experiments",
    )

    segment_group = parser.add_mutually_exclusive_group(required=False)
    segment_group.add_argument(
        "--num_segments",
        type=int,
        default=None,
        help="number of the segments to split the rollouts into (default: None); "
        "either use num_segments or len_segments but not both",
    )
    segment_group.add_argument(
        "--len_segments",
        type=int,
        default=None,
        help="length of the segments to split the rollouts into (default: None); "
        "either use len_segments or num_segments but not both",
    )

    return parser
