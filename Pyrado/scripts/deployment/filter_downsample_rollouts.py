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
Script to filter and downsample recorded rollouts
"""
import os

from matplotlib import pyplot as plt
from scipy import signal

import pyrado
from pyrado.plotting.rollout_based import plot_observations
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_rollouts_from_dir
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument("--factor", type=int, default=1, help="downsampling factor (default: 1, i.e. no downsampling)")
    parser.add_argument(
        "--f_cut", type=float, default=50, help="cutoff frequency of the Butterworth filter in Hz (default: 50Hz)"
    )
    args = parser.parse_args()
    if args.dir is None:
        raise pyrado.ValueErr(msg="Please provide a directory using -d or --dir")
    if args.dt is None:
        raise pyrado.ValueErr(msg="Please provide the time step size used during recoding via --dt")

    # Load the rollouts
    rollouts, file_names = load_rollouts_from_dir(args.dir)

    # Create a lowpass Butterworth filter with a cutoff at 50 Hz, and a sampling frequency of the orig system of 500Hz
    b, a = signal.butter(N=10, Wn=args.f_cut, fs=1 / args.dt)

    for ro, fname in zip(rollouts, file_names):
        ro.numpy()
        if args.verbose:
            plot_observations(ro)
            plt.gcf().canvas.manager.set_window_title("Before")

        # Filter the signals, but not the time
        ro_proc = StepSequence.process_data(
            ro, signal.filtfilt, fcn_arg_name="x", exclude_fields=["time"], b=b, a=a, padlen=150, axis=0
        )

        # Downsample all data fields
        ro_proc = StepSequence.process_data(ro_proc, lambda x: x[:: args.factor], fcn_arg_name="x")

        if args.verbose:
            plot_observations(ro_proc)
            plt.gcf().canvas.manager.set_window_title("After")
            plt.show()

        # Save in a new folder on the same level as the current folder
        curr_dir = args.dir[args.dir.rfind("/") + 1 :]
        save_dir = os.path.join(args.dir, "..", curr_dir + f"_filt_dwnsmp_{args.factor}")
        os.makedirs(save_dir, exist_ok=True)
        pyrado.save(ro_proc, f"{fname}.pkl", save_dir)
        print_cbt(f"Saved {fname}.pkl to {save_dir}", "g")
