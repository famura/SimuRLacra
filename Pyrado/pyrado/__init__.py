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

import numpy as np
import os.path as osp
import random
import torch as to
from colorama import init
from os import _exit
from typing import Union


# Pyrado version number
VERSION = '0.2'

# Provide global data directories
PERMA_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma')
EVAL_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma', 'evaluation')
EXP_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma', 'experiments')
HPARAM_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'perma', 'hyperparams')
TEMP_DIR = osp.join(osp.dirname(__file__), '..', 'data', 'temp')
MUJOCO_ASSETS_DIR = osp.join(osp.dirname(__file__), 'environments', 'mujoco', 'assets')
ISAAC_ASSETS_DIR = osp.join(osp.dirname(__file__), '..', '..', 'thirdParty', 'isaac_gym', 'assets')

# Check if the interfaces to the physics engines are available
try:
    import rcsenv
except ImportError:
    rcsenv_available = False
else:
    rcsenv_available = True

try:
    import mujoco_py
except (ImportError, Exception):
    # The ImportError is raised if mujoco-py is simply not installed
    # The Exception catches the case that you have everything installed properly but your IDE does not set the
    # LD_LIBRARY_PATH correctly (happens for PyCharm & CLion). To check this, try to run your script from the terminal.
    mujoco_available = False
else:
    mujoco_available = True

# Set default data type for PyTorch
to.set_default_dtype(to.double)

# Convenient math variables
inf = float('inf')
nan = float('nan')

# Figure sizes (width, height) [inch]; measures are taken w.r.t. the document's line length
figsize_thesis_1percol_18to10 = (5.8, 5.8/18*10)
figsize_thesis_1percol_16to10 = (5.8, 5.8/16*10)
figsize_thesis_2percol_18to10 = (2.9, 2.9/18*10)
figsize_thesis_2percol_16to10 = (2.9, 2.9/16*10)
figsize_thesis_2percol_square = (2.9, 2.9)
figsize_IEEE_1col_18to10 = (3.5, 3.5/18*10)
figsize_IEEE_2col_18to10 = (7.16, 7.16/18*10)
figsize_IEEE_1col_square = (3.5, 3.5)
figsize_IEEE_2col_square = (7.16, 7.16)
figsize_JMLR_warpfig = (2.5, 2.4)

# Set style for printing and plotting
use_pgf = False
from pyrado import plotting


# Reset the colorama style after each print
init(autoreset=True)

# Set a uniform printing style for PyTorch
to.set_printoptions(precision=4, linewidth=200)

# Set a uniform printing style for numpy
np.set_printoptions(precision=4, sign=' ', linewidth=200)  # suppress=True

sym_success = '\u2714'
sym_failure = '\u2716'

# Include all error classes
from pyrado.utils.exceptions import BaseErr, KeyErr, PathErr, ShapeErr, TypeErr, ValueErr


# Set the public API
__all__ = ['VERSION', 'TEMP_DIR', 'PERMA_DIR', 'EVAL_DIR', 'EXP_DIR', 'HPARAM_DIR',
           'rcsenv_available', 'mujoco_available', 'use_pgf', 'inf', 'nan',
           'sym_success', 'sym_failure']


def close_vpython():
    """ Forcefully close the connection to the current VPython animation """
    _exit(0)


def set_seed(seed: Union[int, None], verbose: bool = False):
    """
    Set the seed for the random number generators

    :param seed: value for the random number generators' seeds
    :param verbose: if `True` the seed is printed
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        to.manual_seed(seed)
        if to.cuda.is_available():
            to.cuda.manual_seed_all(seed)
        if verbose:
            print(f"Set the random number generators' seed to {seed}.")
    else:
        if verbose:
            print(f"The random number generators' seeds were not set.")
