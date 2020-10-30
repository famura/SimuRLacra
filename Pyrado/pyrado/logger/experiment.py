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

import itertools
import numpy as np
import os
import os.path as osp
import torch as to
import torch.nn as nn
import yaml
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Sequence, Union, Iterable, List

import pyrado
from pyrado.logger import set_log_prefix_dir
from pyrado.utils import get_class_name
from pyrado.utils.input_output import select_query, print_cbt


timestamp_format = '%Y-%m-%d_%H-%M-%S'
timestamp_date_format = '%Y-%m-%d'


class Experiment:
    """
    Class for defining experiments
    This is a path-like object, and as such it can be used everywhere a normal path would be used.

    Experiment folder path:
    <base_dir>/<env_name>/<algo_name>/<timestamp>--<extra_info>
    """

    def __init__(self,
                 env_name: str,
                 algo_name: str,
                 extra_info: str = None,
                 exp_id: str = None,
                 timestamp: datetime = None,
                 base_dir: str = pyrado.TEMP_DIR):
        """
        Constructor

        :param env_name: environment trained on
        :param algo_name: algorithm trained with, usually also includes the policy type, e.g. 'a2c_fnn'
        :param extra_info: additional information on the experiment (free form)
        :param exp_id: combined timestamp and extra_info, usually the final folder name
        :param timestamp: experiment creation timestamp
        :param base_dir: base storage directory
        """
        if exp_id is not None:
            # Try to parse extra_info from exp id
            sd = exp_id.split('--', 1)
            if len(sd) == 1:
                timestr = sd[0]
            else:
                timestr, extra_info = sd
            # Parse time string
            if '_' in timestr:
                timestamp = datetime.strptime(timestr, timestamp_format)
            else:
                timestamp = datetime.strptime(timestr, timestamp_date_format)
        else:
            # Create exp id from timestamp and info
            if timestamp is None:
                timestamp = datetime.now()
            exp_id = timestamp.strftime(timestamp_format)

            if extra_info is not None:
                exp_id = exp_id + '--' + extra_info

        # Store values
        self.env_name = env_name
        self.algo_name = algo_name
        self.extra_info = extra_info
        self.exp_id = exp_id
        self.timestamp = timestamp
        self.base_dir = base_dir

    def __fspath__(self):
        """ Allows to use the experiment object where the experiment path is needed. """
        return osp.join(self.base_dir, self.env_name, self.algo_name, self.exp_id)

    def __str__(self):
        """ Get an information string. """
        return f'{self.env_name}/{self.algo_name}/{self.exp_id}'

    @property
    def prefix(self):
        """ Combination of experiment and algorithm """
        return osp.join(self.env_name, self.algo_name)

    def matches(self, hint: str) -> bool:
        """ Check if this experiment matches the given hint. """
        # Split hint into <env>/<algo>/<id>
        parts = Path(hint).parts
        if len(parts) == 1:
            # Filter by exp name only
            env_name, = parts
            return self.env_name == env_name
        elif len(parts) == 2:
            # Filter by exp name only
            env_name, algo_name = parts
            return self.env_name == env_name and self.algo_name == algo_name
        elif len(parts) == 3:
            # Filter by exp name only
            env_name, algo_name, eid = parts
            return self.env_name == env_name and self.algo_name == algo_name and self.exp_id == eid


def setup_experiment(env_name: str,
                     algo_name: str,
                     extra_info: str = None,
                     base_dir: str = pyrado.TEMP_DIR):
    """ Setup a new experiment for recording. """
    # Create experiment object
    exp = Experiment(env_name, algo_name, extra_info, base_dir=base_dir)

    # Create the folder
    os.makedirs(exp, exist_ok=True)

    # Set the global logger variable
    set_log_prefix_dir(exp)

    return exp


# Only child directories
def _childdirs(parent: str):
    for cn in os.listdir(parent):
        cp = osp.join(parent, cn)
        if osp.isdir(cp):
            yield cn


def _le_env_algo(env_name: str, algo_name: str, base_dir: str):
    for exp_id in _childdirs(osp.join(base_dir, env_name, algo_name)):
        yield Experiment(env_name, algo_name, exp_id=exp_id, base_dir=base_dir)


def _le_env(env_name: str, base_dir: str):
    for algo_name in _childdirs(osp.join(base_dir, env_name)):
        yield from _le_env_algo(env_name, algo_name, base_dir)


def _le_base(base_dir: str):
    for env_name in _childdirs(base_dir):
        yield from _le_env(env_name, base_dir)


def _le_select_filter(env_name: str, algo_name: str, base_dir: str):
    if env_name is None:
        return _le_base(base_dir)
    if algo_name is None:
        return _le_env(env_name, base_dir)
    return _le_env_algo(env_name, algo_name, base_dir)


def list_experiments(env_name: str = None,
                     algo_name: str = None,
                     base_dir: str = None,
                     *,
                     temp: bool = True,
                     perma: bool = True):
    """
    List all stored experiments.

    :param env_name: filter by env name
    :param algo_name: filter by algorithm name. Requires env_name to be used too
    :param base_dir: explicit base dir if desired. May also be a list of bases. Uses temp and perm dir if not specified.
    :param temp: set to `False` to not look in the `pyrado.TEMP` directory
    :param perma: set to `False` to not look in the `pyrado.PERMA` directory
    """
    # Parse bases
    if base_dir is None:
        # Use temp/perm if requested
        if temp:
            yield from _le_select_filter(env_name, algo_name, pyrado.TEMP_DIR)
        if perma:
            yield from _le_select_filter(env_name, algo_name, pyrado.EXP_DIR)
    elif not isinstance(base_dir, (str, bytes, os.PathLike)):
        # Multiple base dirs
        for bd in base_dir:
            yield from _le_select_filter(env_name, algo_name, bd)
    else:
        # Single base dir
        yield from _le_select_filter(env_name, algo_name, base_dir)


def _select_latest(exps: Iterable) -> Union[Experiment, None]:
    """
    Select the most recent experiment from an iterable of experiments. Return `None` if there are no experiments.

    :param exps: iterable of experiments
    :return: latest experiment ot `None`
    """
    se = sorted(exps, key=lambda exp: exp.timestamp, reverse=True)  # sort from latest to oldest
    return None if len(se) == 0 else se[0]


def _select_all(exps: Iterable) -> Union[List[Experiment], None]:
    """
    Select all experiments from an iterable of experiments and sort them from from latest to oldest.
    Return `None` if there are no experiments.

    :param exps: iterable of experiments
    :return: temporally sorted experiment ot `None`
    """
    se = sorted(exps, key=lambda exp: exp.timestamp, reverse=True)  # sort from latest to oldest
    return None if len(se) == 0 else se


def select_by_hint(exps: Sequence[Experiment], hint: str):
    """ Select experiment by hint. """
    if osp.isabs(hint):
        # Hint is a full experiment path
        return hint

    # Select matching exps
    selected = filter(lambda exp: exp.matches(hint), exps)
    sl = _select_latest(selected)

    if sl is None:
        print_cbt(f'No experiment matching hint {hint}', 'r')
    return sl


def split_path_custom_common(path: Union[str, Experiment]) -> (str, str):
    """
    Split a path at the point where the machine-dependent and the machine-independent part can be separated.

    :param path: (complete) experiment path to be split
    :return: name of the base directory ('experiments' for `pyrado.EXP_DIR` or 'temp' for `pyrado.TEMP_DIR`) where the
             experiment was located, and machine-independent part of the path
    """

    def _split_path_at(path, key):
        """
        Split a path at the point where the machine-dependent and the machine-independent part can be separated.
        In general, the paths look like this
        `/CUSTOM_FOR_EVERY_MACHINE/SimuRLacra/Pyrado/pyrado/../data/CUSTOM_FOR_EVERY_EXPERIMENT'`
        Thus, we look for the first occurrence of the word 'data'.

        :param path: (complete) experiment path to be split
        :return: part of the path until 'data/, and machine-independent part of the path
        """
        if isinstance(path, Experiment):
            path = os.fspath(path)  # convert Experiment to PathLike a.k.a. string
        idx = path.find(key + '/')
        if idx == -1:
            raise pyrado.PathErr(msg='The key word was not found in the path!')
        else:
            idx += len(key) + 1  # +1 for the '/'
        return path[:idx], path[idx:]

    try:
        return _split_path_at(path, key='experiments')
    except pyrado.PathErr:
        return _split_path_at(path, key='temp')


def ask_for_experiment(latest_only: bool = False):
    """
    Ask for an experiment on the console. This is the go-to entry point for evaluation scripts.

    :param latest_only: only select the latest experiment of each type (environment-algorithm combination)
    :return: query asking the user for an experiment
    """
    # Scan for experiment list
    all_exps = list(list_experiments())

    if len(all_exps) == 0:
        print_cbt('No experiments found!', 'r')
        exit(1)

    # Obtain experiment prefixes and timestamps
    all_exps.sort(key=lambda exp: exp.prefix)  # sorting experiments from early to late
    exps_by_prefix = itertools.groupby(all_exps, key=lambda exp: exp.prefix)  # grouping by env-algo combination
    if latest_only:
        sel_exp_by_prefix = [_select_latest(exps) for _, exps in exps_by_prefix]
    else:
        sel_exp_by_prefix = [_select_all(exps) for _, exps in exps_by_prefix]
        sel_exp_by_prefix = list(itertools.chain.from_iterable(sel_exp_by_prefix))  # flatten list of lists
    sel_exp_by_prefix.sort(key=lambda exp: exp.timestamp, reverse=True)

    # Ask nicely
    return select_query(
        sel_exp_by_prefix,
        fallback=lambda hint: select_by_hint(all_exps, hint),
        item_formatter=lambda exp: f'({exp.timestamp}) {exp.prefix}',
        header='Available experiments:',
        footer='Enter experiment number or a partial path to an experiment.'
    )


def _process_list_for_saving(l: list) -> list:
    """
    The yaml.dump function can't save Tensors, ndarrays, or callables, so we cast them to types it can save.

    :param l: list containing parameters to save
    :return: list with values processable by yaml.dump
    """
    copy = deepcopy(l)  # do not mutate the input
    for i, item in enumerate(copy):
        # Check the values of the list
        if isinstance(item, (to.Tensor, np.ndarray)):
            # Save Tensors as lists
            copy[i] = item.tolist()
        elif isinstance(item, np.float64):
            # PyYAML can not save numpy floats
            copy[i] = float(item)
        elif isinstance(item, nn.Module):
            # Only save the class name as a sting
            copy[i] = get_class_name(item)
        elif callable(item):
            # Only save function name as a sting
            try:
                copy[i] = str(item)
            except AttributeError:
                copy[i] = item.__name__
        elif isinstance(item, dict):
            # If the value is another dict, recursively go through this one
            copy[i] = _process_dict_for_saving(item)
        elif isinstance(item, list):
            # If the value is a list, recursively go through this one
            copy[i] = _process_list_for_saving(item)
        elif item is None:
            copy[i] = 'None'
    return copy


def _process_dict_for_saving(d: dict) -> dict:
    """
    The yaml.dump function can't save Tensors, ndarrays, or callables, so we cast them to types it can save.

    :param d: dict containing parameters to save
    :return: dict with values processable by yaml.dump
    """
    copy = deepcopy(d)  # do not mutate the input
    for k, v in copy.items():
        # Check the values of the dict
        if isinstance(v, (to.Tensor, np.ndarray)):
            # Save Tensors as lists
            copy[k] = v.tolist()
        elif isinstance(v, np.float64):
            # PyYAML can not save numpy floats
            copy[k] = float(v)
        elif isinstance(v, nn.Module):
            # Only save the class name as a sting
            copy[k] = get_class_name(v)
        elif callable(v):
            # Only save function name as a sting
            try:
                copy[k] = str(v)
            except AttributeError:
                try:
                    copy[k] = get_class_name(v)
                except Exception:
                    copy[k] = v.__name__
        elif isinstance(v, dict):
            # If the value is another dict, recursively go through this one
            copy[k] = _process_dict_for_saving(v)
        elif isinstance(v, list):
            # If the value is a list, recursively go through this one
            copy[k] = _process_list_for_saving(v)
        elif v is None:
            copy[k] = 'None'
    return copy


class AugmentedSafeLoader(yaml.SafeLoader):
    def construct_python_tuple(self, node):
        """ Use PyYAML method for constructing a sequence to construct a tuple. """
        return tuple(self.construct_sequence(node))


AugmentedSafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    AugmentedSafeLoader.construct_python_tuple)


def save_list_of_dicts_to_yaml(lod: Sequence[dict], save_dir: str, file_name: str = 'hyperparams'):
    """
    Save a list of dicts (e.g. hyper-parameters) of an experiment a YAML-file.

    :param lod: list of dicts each containing 1 key (name) and 1 value (a dict with the hyper-parameters)
    :param save_dir: directory to save the results in
    :param file_name: name of the YAML-file without suffix
    """
    with open(osp.join(save_dir, file_name + '.yaml'), 'w') as yaml_file:
        for d in lod:
            # For every dict in the list
            d = _process_dict_for_saving(d)
            yaml.dump(d, yaml_file, default_flow_style=False, allow_unicode=True)


def load_dict_from_yaml(yaml_file: str) -> dict:
    """
    Load a list of dicts (e.g. hyper-parameters) of an experiment from a YAML-file.

    :param yaml_file: path to the YAML-file that
    :return: a dict containing names as keys and a dict of parameter values
    """
    if not osp.isfile(yaml_file):
        raise pyrado.PathErr(given=yaml_file)

    with open(yaml_file, 'r') as yaml_file:
        data = yaml.load(yaml_file, Loader=AugmentedSafeLoader)
    return data
