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

import joblib
import numpy as np
import torch as to
from os import path as osp
from typing import Optional

from pyrado.utils.exceptions import PathErr, TypeErr, ValueErr


def _save_fcn(obj, path, extension):
    """ Actual saving function, which handles the cases specified in `save()`. """
    if extension == 'pt':
        to.save(obj, path)
    elif extension == 'npy':
        np.save(path, obj)
    elif extension == 'pkl':
        joblib.dump(obj, path)
    else:
        return NotImplementedError


def _load_fcn(path, extension):
    """ Actual loading function, which handles the cases specified in `load()`. """
    if extension == 'pt':
        obj = to.load(path)
    elif extension == 'npy':
        obj = np.load(path)
    elif extension == 'pkl':
        obj = joblib.load(path)
    else:
        return NotImplementedError
    return obj


def save(obj,
         name: str,
         file_ext: str,
         save_dir: str,
         meta_info: Optional[dict] = None,
         use_state_dict: bool = True):
    """
    Save an object object using a prefix or suffix, depending on the meta information.

    :param obj: PyTorch or pickled object to save
    :param name: name of the object for saving
    :param file_ext: file extension, extension.g. 'pt' for PyTorch modules like the Pyrado policies
    :param save_dir: directory to save in
    :param meta_info: meta information that can contain a pre- and/or suffix for altering the name
    :param use_state_dict: if `True` save the `state_dict`, else save the entire module. This only has an effect if
                           PyTorch modules (file_ext = 'pt') are saved.

    .. seealso::
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    """
    if not isinstance(name, str):
        raise TypeErr(given=name, expected_type=str)
    if not (file_ext in ['pt', 'npy', 'pkl']):
        raise ValueErr(given=file_ext, eq_constraint='pt, npy, or pkl')
    if not osp.isdir(save_dir):
        raise PathErr(given=save_dir)

    if file_ext == 'pt' and use_state_dict:
        # Later save the model's sate dict if possible. If not, save the entire object
        if hasattr(obj, 'state_dict'):
            obj_ = obj.state_dict()
        else:
            obj_ = obj
    else:
        # Later save (and pickle) the entire model
        obj_ = obj

    if meta_info is None:
        _save_fcn(obj_, osp.join(save_dir, f"{name}.{file_ext}"), file_ext)

    else:
        if not isinstance(meta_info, dict):
            raise TypeErr(given=meta_info, expected_type=dict)

        if 'prefix' in meta_info and 'suffix' in meta_info:
            _save_fcn(obj_, osp.join(save_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"),
                      file_ext)

        elif 'prefix' in meta_info and 'suffix' not in meta_info:
            _save_fcn(obj_, osp.join(save_dir, f"{meta_info['prefix']}_{name}.{file_ext}"), file_ext)

        elif 'prefix' not in meta_info and 'suffix' in meta_info:
            _save_fcn(obj_, osp.join(save_dir, f"{name}_{meta_info['suffix']}.{file_ext}"), file_ext)

        else:  # there is meta_info dict but with different key words
            _save_fcn(obj_, osp.join(save_dir, f"{name}.{file_ext}"), file_ext)


def load(obj, name: str, file_ext: str, load_dir: str, meta_info: Optional[dict] = None):
    """
    Load an object object using a prefix or suffix, depending on the meta information.
    
    :param obj: PyTorch modeule to load into, this can be `None` except for the case if you want to load and save the
                module's `state_dict`
    :param name: name of the object for loading
    :param file_ext: file extension, extension.g. 'pt' for PyTorch modules like the Pyrado policies
    :param load_dir: directory to load from
    :param meta_info: meta information that can contain a pre- and/or suffix for altering the name
    
    .. seealso::
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    """
    if not isinstance(name, str):
        raise TypeErr(given=name, expected_type=str)
    if not (file_ext in ['pt', 'npy', 'pkl']):
        raise ValueErr(given=file_ext, eq_constraint='pt, npy, or pkl')
    if not osp.isdir(load_dir):
        raise PathErr(given=load_dir)

    if meta_info is None:
        obj_ = _load_fcn(osp.join(load_dir, f"{name}.{file_ext}"), file_ext)

    else:
        if not isinstance(meta_info, dict):
            raise TypeErr(given=meta_info, expected_type=dict)

        if 'prefix' in meta_info and 'suffix' in meta_info:
            obj_ = _load_fcn(osp.join(load_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"),
                             file_ext)

        elif 'prefix' in meta_info and 'suffix' not in meta_info:
            obj_ = _load_fcn(osp.join(load_dir, f"{meta_info['prefix']}_{name}.{file_ext}"), file_ext)

        elif 'prefix' not in meta_info and 'suffix' in meta_info:
            obj_ = _load_fcn(osp.join(load_dir, f"{name}_{meta_info['suffix']}.{file_ext}"), file_ext)

        else:  # there is meta_info dict but with different key words
            obj_ = _load_fcn(osp.join(load_dir, f"{name}.{file_ext}"), file_ext)

    assert obj_ is not None
    if isinstance(obj_, dict) and file_ext == 'pt':
        # PyTorch saves state_dict as an OrderedDict
        obj.load_state_dict(obj_)
    else:
        obj = obj_
    return obj
