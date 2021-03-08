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

import collections
import numpy as np
import torch as to
import typing
from typing import Sequence, NamedTuple, Union, TypeVar

import pyrado
import typing
from pyrado.spaces.base import Space
from pyrado.spaces.empty import EmptySpace
from pyrado.utils.checks import is_sequence

T = TypeVar("T")


class EnvSpec(NamedTuple):
    """ The specification of an environment's input and output space """

    obs_space: Space
    act_space: Space
    state_space: Space = EmptySpace()


class RenderMode(NamedTuple):
    """ The specification of a render mode, do not print or render anything by default """

    text: bool = False
    video: bool = False
    render: bool = False


class TimeSeriesDataPair(NamedTuple):
    """ Pair of an input sequence and an associated target value for training time series prediction. """

    inp_seq: to.Tensor
    targ: to.Tensor


class DSSpec(dict):
    """
    The specification of a dynamical system implemented in `RcsPySim` and used in combination with ADN
    We are directly subclassing dict beacuse `PropertySource` in `RcsPySim` extracts information from dicts.
    """

    def __init__(self, function: str, goal: Union[np.ndarray, float, int], **kwargs):
        """
        Constructor

        :param function: name of the dynamics function, e.g. `linear`, `msd`, or `msd_nlin`
        :param goal: desired state, e.g. a task space position or velocity
        :param kwargs: additional attributes set by subclasses
        """
        # Call dict's constructor since we want instances of this class to be accessible like a dict
        super().__init__(dict(function=function, goal=goal, **kwargs))


class MSDDSSpec(DSSpec):
    """
    The specification of a second order dynamical system i.e. a linear or nonlinear mass spring damper system
    implemented in `RcsPySim`.
    Example:
    MSDDSSpec(function='msd_nlin', goal=np.array([-1., 2.]), attractorStiffness=50., damping=10., mass=1.)
    """

    def __init__(
        self,
        function: str,
        goal: Union[np.ndarray, float, int],
        attractorStiffness: Union[float, int],
        damping: Union[float, int],
        mass: Union[float, int] = 1.0,
    ):
        """
        Constructor

        :param function: name of the dynamics function, e.g. `msd`, or `msd_nlin`
        :param goal: desired state, e.g. a task space position or velocity
        :param attractorStiffness: spring stiffness parameter of the dynamical system
        :param damping: damping parameter of the dynamical system
        :param mass: mass parameter of the dynamical system
        """
        if not (function == "msd" or function == "msd_nlin"):
            raise pyrado.ValueErr(given=function, eq_constraint="'msd' or 'msd_nlin")

        super().__init__(
            function=function, goal=goal, attractorStiffness=attractorStiffness, damping=damping, mass=mass
        )


class LinDSSpec(DSSpec):
    r"""
    The specification of a linear first order dynamical system implemented in `RcsPySim`.
    The dynamics are defined as $\dot{x} = A (x_{des} - x_{curr})$ with the error dynamics matrix $A$.
    Example:
    LinDSSpec(function='lin', goal=np.array([-1., 2.]), errorDynamics=np.eye(2))
    """

    def __init__(
        self, function: str, goal: Union[np.ndarray, float, int], errorDynamics: Union[np.ndarray, float, int]
    ):
        """
        Constructor

        :param function: name of the dynamics function, e.g. `msd`, or `msd_nlin`
        :param goal: desired state, e.g. a task space position or velocity
        :param errorDynamics: a matrix determining the dynamics or the error difference between
        """
        if not function == "lin":
            raise pyrado.ValueErr(given=function, eq_constraint="'lin'")

        super().__init__(function=function, goal=goal, errorDynamics=errorDynamics)


def repeat_interleave(sequence: Union[list, tuple], num_reps) -> Union[list, tuple]:
    """
    Repeat every element of the input sequence, but keep the order of the elements.

    :param sequence: input list or tuple
    :param num_reps: number of repetitions
    :return: list or tuple with repeated elements
    """
    if isinstance(sequence, list):
        return [ele for ele in sequence for _ in range(num_reps)]
    elif isinstance(sequence, tuple):
        return tuple(ele for ele in sequence for _ in range(num_reps))
    else:
        raise pyrado.TypeErr(given=sequence, expected_type=(list, tuple))


def merge_dicts(dicts: Sequence[dict]) -> dict:
    """
    Marge a list of dicts in to one dict.
    The resulting dict can have different value types, but the dicts later in the list override earlier ones.

    :param dicts: input dicts
    :return: merged dict
    """
    if not is_sequence(dicts):
        raise pyrado.TypeErr(given=dicts, expected_type=Sequence)

    merged = {}
    for d in dicts:
        if d is not None:  # skip None entries to use this with kwargs
            if not isinstance(d, dict):
                raise pyrado.TypeErr(given=d, expected_type=dict)
            merged.update(d)
    return merged


def merge_dicts_same_dtype(dicts: Sequence[dict], dtype=set) -> dict:
    """
    Marge a list of dicts in to one dict.
    The resulting dict has only one value type for all entries, but the dicts do not override each other.

    :param dicts: input dicts
    :param dtype: data type for constructing the default dict
    :return: merged dict
    """
    if not is_sequence(dicts):
        raise pyrado.TypeErr(given=dicts, expected_type=Sequence)

    merged = collections.defaultdict(dtype)
    for d in dicts:
        if d is not None:  # skip None entries to use this with kwargs
            if not isinstance(d, dict):
                raise pyrado.TypeErr(given=d, expected_type=dict)
            for k, v in d.items():
                merged[k].add(v)
    return merged


def to_float_relentless(inp):
    """
    Convert a ndarray to a list of floats or to only one float.
    Convert a list of numerical values to a list of floats to only one float.
    Convert a scalar to a float.
    :param inp: ndarray, or list, or scalar
    :return: scalar number or list of floats
    """
    if isinstance(inp, np.ndarray):
        if np.isscalar(inp) or inp.size == 1:
            return float(inp.item())
        else:
            return inp.astype(dtype=float).tolist()
    elif isinstance(inp, list):
        for i, x in enumerate(inp):
            try:
                inp[i] = float(x)
            except Exception:
                pass
        return inp
    else:
        # Relentless
        try:
            inp = float(inp)
        except Exception:
            pass
        return inp


def dict_arraylike_to_float(d: dict):
    """
    Convert every scalar array-like entry the provided dict to to a float.
    This function is useful when the metrics (e.g., calculated with numpy) should be saved in a YAML-file.
    :param d: input dict with 1-element arrays
    :return d: output dict with float entries where conversion was possible
    """
    for k, v in d.items():
        d[k] = to_float_relentless(v)
    return d


def fill_list_of_arrays(loa: Sequence[np.ndarray], des_len: int, fill_ele=np.nan) -> Sequence[np.ndarray]:
    """
    Fill a list of arrays with potential unequal length (first axis) will provided elements

    :param loa: list of ndarrays
    :param des_len: desired length of the resulting arrays
    :param fill_ele: element to fill the arrays with
    :return: list of ndarrays with equal length
    """
    # Copy the list to avoid undesired changes to the input
    loa_c = list(loa)
    if not isinstance(loa_c, (tuple, list)):
        raise pyrado.TypeErr(given=loa_c, expected_type=[tuple, list])

    # Go through the list
    for i, a in enumerate(loa):
        # Loop over rollouts anc check their size
        if a.shape[0] < des_len:
            # Create fill array if a is multi-dim
            fill_arr = np.empty((1, a.shape[1]))
            fill_arr[:] = fill_ele
            # Append fill_ele until the array is of length des_len
            loa_c[i] = np.append(a, np.repeat(fill_arr, des_len - a.shape[0], axis=0), axis=0)

    # Return the modified copy
    return loa_c


def dict_path_access(
    d: dict, path: str, default: typing.Optional[T] = None, default_for_last_layer_only: bool = False
) -> T:
    """
    Gets a dict entry using a jq-like naming system. To get the entry `foo` from the topmost
    dictionary, use the path `foo`. To access an entry `baz` in a dictionary that is stored in the
    top dictionary under the key `bar`, use the path `bar.baz`.

    :param d: the dictionary
    :param path: the path
    :param default: if no entry corresponding to the key was found, this value is returned
    :param default_for_last_layer_only: if `True`, only return the default value if the entry was
                                        not found in the last dictionary, but the previous
                                        dictionaries where found (e.g. for `bar.baz`, `d['bar']`
                                        exists, but `d['bar']['baz']` does not; if those where not
                                        found, an error is raised; if `False`, also return the
                                        default value of non-existent dictionaries along the path
    :return: the entry, if found; if any part of the path except for the last does not exist, and
             `default_for_last_layer_only` is `True`, an error is raised; in all other cases, the
             default value is returned
    """

    result = d
    path_split = path.split(".")
    for i, part in enumerate(path_split):
        if part not in result and (i == len(path_split) - 1 or not default_for_last_layer_only):
            return default
        result = result[part]
    return result
