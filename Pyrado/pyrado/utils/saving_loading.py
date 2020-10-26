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

import pyrado
from pyrado.utils import get_class_name
from pyrado.utils.input_output import print_cbt


def save_prefix_suffix(obj,
                       name: str,
                       file_ext: str,
                       save_dir: str,
                       meta_info: Optional[dict] = None,
                       use_state_dict: bool = True):
    """
    Save an object object using a prefix or suffix, depending on the meta information.

    :param obj: PyTorch or pickled object to save
    :param name: name of the object for saving
    :param file_ext: file extension, e.g. 'pt' for PyTorch modules like the Pyrado policies
    :param save_dir: directory to save in
    :param meta_info: meta information that can contain a pre- and/or suffix for altering the name
    :param use_state_dict: if `True` save the `state_dict`, else save the entire module. This only has an effect if
                           PyTorch modules (file_ext = 'pt') are saved.

    .. seealso::
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    """
    if not isinstance(name, str):
        raise pyrado.TypeErr(given=name, expected_type=str)
    if not (file_ext in ['pt', 'npy', 'pkl']):
        raise pyrado.ValueErr(given=file_ext, eq_constraint='pt, npy, or pkl')
    if not osp.isdir(save_dir):
        raise pyrado.PathErr(given=save_dir)

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
        if file_ext == 'pt':
            to.save(obj_, osp.join(save_dir, f"{name}.{file_ext}"))

        elif file_ext == 'npy':
            np.save(osp.join(save_dir, f"{name}.{file_ext}"), obj_)

        elif file_ext == 'pkl':
            joblib.dump(obj_, osp.join(save_dir, f"{name}.{file_ext}"))

    else:
        if not isinstance(meta_info, dict):
            raise pyrado.TypeErr(given=meta_info, expected_type=dict)

        if file_ext == 'pt':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                to.save(obj_,
                        osp.join(save_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                to.save(obj_, osp.join(save_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                to.save(obj_, osp.join(save_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                to.save(obj_, osp.join(save_dir, f"{name}.{file_ext}"))

        elif file_ext == 'npy':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                np.save(osp.join(save_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"), obj_)
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                np.save(osp.join(save_dir, f"{meta_info['prefix']}_{name}.{file_ext}"), obj_)
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                np.save(osp.join(save_dir, f"{name}_{meta_info['suffix']}.{file_ext}"), obj_)
            else:
                np.save(osp.join(save_dir, f"{name}.{file_ext}"), obj_)

        elif file_ext == 'pkl':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                joblib.dump(obj_, osp.join(save_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                joblib.dump(obj_, osp.join(save_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                joblib.dump(obj_, osp.join(save_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                joblib.dump(obj_, osp.join(save_dir, f"{name}.{file_ext}"))


def load_prefix_suffix(obj, name: str, file_ext: str, load_dir: str, meta_info: Optional[dict] = None):
    """
    Load an object object using a prefix or suffix, depending on the meta information.

    :param obj: PyTorch modeule to load into, this can be `None` except for the case if you want to load and save the
                module's `state_dict`
    :param name: name of the object for loading
    :param file_ext: file extension, e.g. 'pt' for PyTorch modules like the Pyrado policies
    :param load_dir: directory to load from
    :param meta_info: meta information that can contain a pre- and/or suffix for altering the name

    .. seealso::
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    """
    if not isinstance(name, str):
        raise pyrado.TypeErr(given=name, expected_type=str)
    if not (file_ext in ['pt', 'npy', 'pkl']):
        raise pyrado.ValueErr(given=file_ext, eq_constraint='pt, npy, or pkl')
    if not osp.isdir(load_dir):
        raise pyrado.PathErr(given=load_dir)

    obj_ = None
    if meta_info is None:
        if file_ext == 'pt':
            obj_ = to.load(osp.join(load_dir, f"{name}.{file_ext}"))

        elif file_ext == 'npy':
            obj_ = np.load(osp.join(load_dir, f"{name}.{file_ext}"))

        elif file_ext == 'pkl':
            obj_ = joblib.load(osp.join(load_dir, f"{name}.{file_ext}"))

    else:
        if not isinstance(meta_info, dict):
            raise pyrado.TypeErr(given=meta_info, expected_type=dict)

        if file_ext == 'pt':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                obj_ = to.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                obj_ = to.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                obj_ = to.load(osp.join(load_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                obj_ = to.load(osp.join(load_dir, f"{name}.{file_ext}"))

        if file_ext == 'npy':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                obj_ = np.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                obj_ = np.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                obj_ = np.load(osp.join(load_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                obj_ = np.load(osp.join(load_dir, f"{name}.{file_ext}"))

        if file_ext == 'pkl':
            if 'prefix' in meta_info and 'suffix' in meta_info:
                obj_ = joblib.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}_{meta_info['suffix']}.{file_ext}"))
            elif 'prefix' in meta_info and 'suffix' not in meta_info:
                obj_ = joblib.load(osp.join(load_dir, f"{meta_info['prefix']}_{name}.{file_ext}"))
            elif 'prefix' not in meta_info and 'suffix' in meta_info:
                obj_ = joblib.load(osp.join(load_dir, f"{name}_{meta_info['suffix']}.{file_ext}"))
            else:
                obj_ = joblib.load(osp.join(load_dir, f"{name}.{file_ext}"))

    assert obj_ is not None
    if isinstance(obj_, dict) and file_ext == 'pt':
        # PyTorch saves state_dict as an OrderedDict
        obj.load_state_dict(obj_)
    else:
        obj = obj_
    return obj
