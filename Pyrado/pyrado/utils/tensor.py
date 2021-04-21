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

from copy import deepcopy
from typing import TypeVar, Union

import torch as to

import pyrado


def stack_tensor_list(tensor_list: list) -> to.Tensor:
    """
    Convenience function for stacking a list of tensors

    :param tensor_list: list of tensors to stack (along a new dim)
    :return: tensor of at least 1-dim
    """
    if not to.is_tensor(tensor_list[0]):
        # List of scalars (probably)
        return to.tensor(tensor_list)
    return to.stack(tensor_list)


def stack_tensor_dict_list(tensor_dict_list: list) -> dict:
    """
    Stack a list of dictionaries of {tensors or dict of tensors}.

    :param tensor_dict_list: a list of dicts of {tensors or dict of tensors}.
    :return: a dict of {stacked tensors or dict of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def insert_tensor_col(x: to.Tensor, idx: int, col: to.Tensor) -> to.Tensor:
    """
    Insert a column into a PyTorch Tensor.

    :param x: original tensor
    :param idx: column index where to insert the column
    :param col: tensor to insert
    :return: tensor with new column at index idx
    """
    assert isinstance(x, to.Tensor)
    assert isinstance(idx, int) and -1 <= idx <= x.shape[1]
    assert isinstance(col, to.Tensor)
    if not x.shape[0] == col.shape[0]:
        raise pyrado.ShapeErr(
            msg=f"Number of rows does not match! Original: {x.shape[0]}, column to insert: {col.shape[0]}"
        )

    # Concatenate along columns
    if 0 <= idx < x.shape[1]:
        return to.cat((x[:, :idx], col, x[:, idx:]), dim=1)
    else:
        # idx == -1 or idx == shape[1]
        return to.cat((x, col), dim=1)


CopyType = TypeVar("CopyType", dict, list)


def deepcopy_or_clone(copy_from: CopyType) -> CopyType:
    """
    Unfortunately, deepcopy() only works for leave tensors right now. Thus, we need to iterate throught the input and
    clone the PyTorch tensors where possible.

    :param copy_from: list or dict to copy
    :return: copy of the input where every tensor was cloned, and every other data type was deepcopied
    """
    if isinstance(copy_from, dict):
        copy = dict()
        for key, value in copy_from.items():
            if isinstance(value, to.Tensor):
                copy[key] = value.clone()
            else:
                copy[key] = deepcopy(value)

    elif isinstance(copy_from, list):
        copy = []
        for item in copy_from:
            if isinstance(item, (dict, list)):
                copy.append(deepcopy_or_clone(item))
            elif isinstance(item, to.Tensor):
                copy.append(item.clone())
            else:
                copy.append(deepcopy(item))

    else:
        raise pyrado.TypeErr(given=copy_from, expected_type=[dict, list])

    return copy
