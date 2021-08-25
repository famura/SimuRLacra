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

import hashlib
import os
import os.path as osp
import platform
import random
from typing import Optional, TypeVar

import numpy as np
import torch as to
from colorama import init


# Pyrado version number
VERSION = "0.7"

# Provide global data directories
PERMA_DIR = osp.join(osp.dirname(__file__), "..", "data", "perma")
EVAL_DIR = osp.join(osp.dirname(__file__), "..", "data", "perma", "evaluation")
EXP_DIR = osp.join(osp.dirname(__file__), "..", "data", "perma", "experiments")
HPARAM_DIR = osp.join(osp.dirname(__file__), "..", "data", "perma", "hyperparams")
TEMP_DIR = osp.join(osp.dirname(__file__), "..", "data", "temp")
MUJOCO_ASSETS_DIR = osp.join(osp.dirname(__file__), "environments", "mujoco", "assets")
PANDA_ASSETS_DIR = osp.join(osp.dirname(__file__), "environments", "pysim", "assets")
if platform.system() == "Linux":
    RENDER_PIPELINE_DIR = osp.join(osp.dirname(__file__), "..", "..", "thirdParty", "RenderPipeline")
else:
    RENDER_PIPELINE_DIR = osp.join(osp.relpath(__file__), "..", "..", "..", "thirdParty", "RenderPipeline")

# Set the availability of the physics-engine based simulations to False. These are set to True in the respective
# top-level __init__.py files, if they can be imported successfully
rcsenv_loaded = False
mujoco_loaded = False

# Set default data type for PyTorch to float32. Sadly this is not possible for numpy.
to.set_default_dtype(to.float32)

# Reset the colorama style after each print
init(autoreset=True)

# Set a uniform printing style
to.set_printoptions(precision=4, linewidth=200)
np.set_printoptions(precision=4, sign=" ", linewidth=200)

# Public API variables
inf = float("inf")
nan = float("nan")
sym_success = "\u2714"
sym_failure = "\u2716"
PathLike = TypeVar("PathLike", str, bytes, os.PathLike)  # PEP 519

# Include error classes
from pyrado.utils.exceptions import BaseErr, KeyErr, PathErr, ShapeErr, TypeErr, ValueErr

# Include saving and loading functions
from pyrado.utils.saving_loading import load, save


# Set style for printing and plotting
use_pgf = False
from pyrado import plotting


# Figure sizes (width, height) [inch]; measures are taken w.r.t. the document's line length
figsize_thesis_1percol_18to10 = (5.8, 5.8 / 18 * 10)
figsize_thesis_1percol_16to10 = (5.8, 5.8 / 16 * 10)
figsize_thesis_2percol_18to10 = (2.9, 2.9 / 18 * 10)
figsize_thesis_2percol_16to10 = (2.9, 2.9 / 16 * 10)
figsize_thesis_2percol_square = (2.9, 2.9)
figsize_IEEE_1col_18to10 = (3.5, 3.5 / 18 * 10)
figsize_IEEE_2col_18to10 = (7.16, 7.16 / 18 * 10)
figsize_IEEE_1col_square = (3.5, 3.5)
figsize_IEEE_2col_square = (7.16, 7.16)
figsize_JMLR_warpfig = (2.5, 2.4)
figsize_CoRL_6perrow_square = (2.9, 0.8)

# Time and date formats
timestamp_format = "%Y-%m-%d_%H-%M-%S"
timestamp_date_format = "%Y-%m-%d"

# Set the public API
__all__ = [
    "VERSION",
    "TEMP_DIR",
    "PERMA_DIR",
    "EVAL_DIR",
    "EXP_DIR",
    "HPARAM_DIR",
    "MUJOCO_ASSETS_DIR",
    "PANDA_ASSETS_DIR",
    "RENDER_PIPELINE_DIR",
    "rcsenv_loaded",
    "mujoco_loaded",
    "use_pgf",
    "inf",
    "nan",
    "PathLike",
    "sym_success",
    "sym_failure",
    "set_seed",
    "timestamp_format",
    "timestamp_date_format",
]

PYRADO_BASE_SEED = None


def set_seed(
    seed: Optional[int], sub_seed: int = None, sub_sub_seed: int = None, verbose: bool = False
) -> Optional[int]:
    """
    Set the seed for the random number generators. The actual seed is computed from the base seed `seed´, and the first-
    and second-order sub-seeds (`sub_seed` and `sub_sub_seed`, respectively). All of these seeds get concatenated in a
    string which is then MD5-hashed and crushed into a 32-bit integer.

    :param seed: base seed, pass `None` to skip seeding; must be an unsigned 10-bit integer
    :param sub_seed: sub-seed, defaults to zero; must be an unsigned 14-bit integer; overflows will be cast back into
                     the interval by masking, i.e., only the last 14 bits will be kept; underflows are first made
                     positive by taking the absolute value
    :param sub_sub_seed: sub-sub-seed, defaults to zero; must be an unsigned 8-bit integer; underflows will be cast back
                         into the interval by taking the absolute value
    :param verbose: if `True` the seed is printed
    :return: the seed that was set
    """

    global PYRADO_BASE_SEED

    # The better parameter name would be 'base_seed', but keep 'seed' for backward compatibility.
    base_seed = seed
    del seed
    if sub_seed is None:
        sub_seed = 0
    if sub_sub_seed is None:
        sub_sub_seed = 0

    if not isinstance(base_seed, int):
        if verbose:
            print(f"Base seed {base_seed} is not an integer -- the random number generators' seeds were not set.")
        return None

    seed = int(hashlib.md5(f"{base_seed}-{sub_seed}-{sub_sub_seed}".encode()).hexdigest(), 16) % (2 ** 32)

    random.seed(seed)
    np.random.seed(seed)
    to.manual_seed(seed)
    if to.cuda.is_available():
        to.cuda.manual_seed_all(seed)
    PYRADO_BASE_SEED = base_seed

    if verbose:
        print(
            f"Set the random number generators' seed to {seed} (computed from base seed {base_seed}, "
            f"sub-seed {sub_seed}, and sub-sub-seed {sub_sub_seed})."
        )

    return seed


def get_base_seed() -> Optional[int]:
    """Gets the last seed that was set with `pyrado.set_seed`. If no seed was every set, `None`."""
    global PYRADO_BASE_SEED
    return PYRADO_BASE_SEED
