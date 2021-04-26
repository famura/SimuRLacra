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

import os
import os.path as osp
from typing import TypeVar

import joblib
import numpy as np
import torch as to

from pyrado.utils.exceptions import PathErr, TypeErr, ValueErr
from pyrado.utils.input_output import print_cbt


# Repeat definition since we can not import pyrado here
PathLike = TypeVar("PathLike", str, bytes, os.PathLike)  # PEP 519


def _save_fcn(obj, path, extension):
    """Actual saving function, which handles the cases specified in `save()`."""
    if extension == "pt":
        to.save(obj, path)
    elif extension == "npy":
        np.save(path, obj)
    elif extension == "pkl":
        joblib.dump(obj, path)
    else:
        return NotImplementedError


def _load_fcn(path, extension):
    """Actual loading function, which handles the cases specified in `load()`."""
    if extension == "pt":
        obj = to.load(path)
    elif extension == "npy":
        obj = np.load(path)
    elif extension == "pkl":
        obj = joblib.load(path)
    else:
        return NotImplementedError
    return obj


def save(obj, name: str, save_dir: PathLike, prefix: str = "", suffix: str = "", use_state_dict: bool = False):
    """
    Save an object object using a prefix or suffix, depending on the meta information.

    :param obj: PyTorch or pickled object to save
    :param name: name of the object for loading including the file extension, e.g. 'policy.pt' for PyTorch modules
                 like a Pyrado `Policy` instance
    :param save_dir: directory to save in
    :param prefix: prefix for altering the name, e.g. "iter_0_..."
    :param suffix: suffix for altering the name, e.g. "..._ref"
    :param use_state_dict: if `True` save the `state_dict`, else save the entire module. This only has an effect if
                           PyTorch modules (file_ext = 'pt') are saved.

    .. seealso::
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    """
    if not isinstance(name, str):
        raise TypeErr(given=name, expected_type=str)
    if not osp.isdir(save_dir):
        raise PathErr(given=save_dir)
    if not isinstance(prefix, str):
        raise TypeErr(given=prefix, expected_type=str)
    elif prefix != "":
        # A valid non-default prefix was given
        prefix = prefix + "_"
    if not isinstance(suffix, str):
        raise TypeErr(given=suffix, expected_type=str)
    elif suffix != "":
        # A valid non-default prefix was given
        suffix = "_" + suffix

    # Infer file type
    file_ext = name[name.rfind(".") + 1 :]
    if not (file_ext in ["pt", "npy", "pkl"]):
        raise ValueErr(msg="Only pt, npy, and pkl files are currently supported!")

    if file_ext == "pt" and use_state_dict:
        # Later save the model's sate dict if possible. If not, save the entire object
        if hasattr(obj, "state_dict"):
            obj_ = obj.state_dict()
        else:
            obj_ = obj
    else:
        # Later save (and pickle) the entire model
        obj_ = obj

    # Save the data
    name_wo_file_ext = name[: name.find(".")]
    _save_fcn(obj_, osp.join(save_dir, f"{prefix}{name_wo_file_ext}{suffix}.{file_ext}"), file_ext)


def load(name: str, load_dir: PathLike, prefix: str = "", suffix: str = "", obj=None, verbose: bool = False):
    """
    Load an object object using a prefix or suffix, depending on the meta information.

    :param name: name of the object for loading including the file extension, e.g. 'policy.pt' for PyTorch modules
                 like a Pyrado `Policy` instance
    :param load_dir: directory to load from
    :param prefix: prefix for altering the name, e.g. "iter_0_..."
    :param suffix: suffix for altering the name, e.g. "..._ref"
    :param obj: PyTorch module to load into, this can be `None` except for the case if you want to load and save the
                module's `state_dict`
    :param verbose: if `True`, print the path of what has been loaded

    .. seealso::
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    """
    if not isinstance(name, str):
        raise TypeErr(given=name, expected_type=str)
    if not osp.isdir(load_dir):
        raise PathErr(given=load_dir)
    if not isinstance(prefix, str):
        raise TypeErr(given=prefix, expected_type=str)
    elif prefix != "":
        # A valid non-default prefix was given
        prefix = prefix + "_"
    if not isinstance(suffix, str):
        raise TypeErr(given=suffix, expected_type=str)
    elif suffix != "":
        # A valid non-default prefix was given
        suffix = "_" + suffix

    # Infer file type
    file_ext = name[name.rfind(".") + 1 :]
    if not (file_ext in ["pt", "npy", "pkl"]):
        raise ValueErr(msg="Only pt, npy, and pkl files are currently supported!")

    # Load the data
    name_wo_file_ext = name[: name.find(".")]
    name_load = f"{prefix}{name_wo_file_ext}{suffix}.{file_ext}"
    obj_ = _load_fcn(osp.join(load_dir, name_load), file_ext)
    assert obj_ is not None

    if isinstance(obj_, dict) and file_ext == "pt":
        # PyTorch saves state_dict as an OrderedDict
        obj.load_state_dict(obj_)
    else:
        obj = obj_

    if verbose:
        print_cbt(f"Loaded {osp.join(load_dir, name_load)}", "g")

    return obj
