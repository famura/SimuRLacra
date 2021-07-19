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
Script to load the data from a real-world rollouts, written to a file by the RcsPySim DAtaLogger class.
"""
import os.path as osp

import pandas as pd

import pyrado
from pyrado.environments.rcspysim.mini_golf import MiniGolfIKSim, MiniGolfJointCtrlSim
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if not osp.isfile(args.file):
        raise pyrado.PathErr(given=args.file)
    if args.dir is None:
        # Use the file's directory by default
        args.dir = osp.dirname(args.file)
    elif not osp.isdir(args.dir):
        raise pyrado.PathErr(given=args.dir)

    df = pd.read_csv(args.file)

    if args.env_name == MiniGolfIKSim.name:
        env = MiniGolfIKSim()
    elif args.env_name == MiniGolfJointCtrlSim.name:
        env = MiniGolfJointCtrlSim()
    else:
        raise NotImplementedError

    # Cast the rollout from a DataFrame to a StepSequence
    reconstructed = StepSequence.from_pandas(df, env.spec, task=env.task)

    if args.dir is not None:
        suffix = args.file[args.file.rfind("/") + 1 : -4]
        pyrado.save(reconstructed, f"rollout_{suffix}.pkl", args.dir, verbose=True)
