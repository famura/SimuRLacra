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
Script to copy and rename an experiment. This is not straight-forward since renaming breaks pickling, thus we need
to run this script, instead of simply renaming the experiment folder.
In case you renamed that folder already, reset the name and run this script. The old name can be recovered from the
algorithm's pickle file (usually called algo.pkl)

.. usage::
    python rename_experiment --dir PATH_TO_OLD_EXPERIMENT --new_dir PATH_TO_COPY_NEW_EXPERIMENT_TO
"""
import os
import os.path as osp
from distutils.dir_util import copy_tree

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.logger.step import CSVPrinter, TensorBoardPrinter
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--new_dir", type=str, nargs="?", help="path to the directory where the experiment should be saved/moved to"
    )
    args = parser.parse_args()

    if not osp.isdir(args.dir):
        raise pyrado.PathErr(given=args.dir)
    if args.new_dir is None:
        raise pyrado.ValueErr(msg="Provide the path to the new experiment directory using --new_dir")

    # Create the new directory and test it
    os.makedirs(args.new_dir, exist_ok=True)
    if not osp.isdir(args.new_dir):
        raise pyrado.PathErr(given=args.new_dir)

    # Load the old algorithm including the loggers
    algo = Algorithm.load_snapshot(args.dir)

    # Update all entries that contain information about where the experiment is stored
    algo.save_dir = args.new_dir
    for printer in algo.logger.printers:
        if isinstance(printer, CSVPrinter):
            printer.file = osp.join(args.new_dir, printer.file[printer.file.rfind("/") + 1 :])
        elif isinstance(printer, TensorBoardPrinter):
            printer.dir = args.new_dir

    # Copy the complete content
    copy_tree(args.dir, args.new_dir)

    # Save the new algorithm with the updated entries
    algo.save_snapshot()
