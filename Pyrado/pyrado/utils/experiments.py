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
import os
import os.path as osp
import xml.etree.ElementTree as et
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch as to

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import load_hyperparameters
from pyrado.policies.base import Policy
from pyrado.policies.recurrent.adn import (
    pd_capacity_21,
    pd_capacity_21_abs,
    pd_capacity_32,
    pd_capacity_32_abs,
    pd_cubic,
    pd_linear,
)
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.argparser import get_argparser
from pyrado.utils.checks import check_all_types_equal, is_iterable
from pyrado.utils.input_output import print_cbt
from pyrado.utils.ordering import natural_sort


def load_experiment(
    ex_dir: str, args: Any = None
) -> Tuple[Optional[Union[SimEnv, EnvWrapper]], Optional[Policy], Optional[dict]]:
    """
    Load the (training) environment and the policy.
    This helper function first tries to read the hyper-parameters yaml-file in the experiment's directory to infer
    why entities should be loaded. If no file was found, we fall back to some heuristic and hope for the best.

    :param ex_dir: experiment's parent directory
    :param args: arguments from the argument parser, pass `None` to fall back to the values from the default argparser
    :return: environment, policy, and (optional) algorithm-specific output, e.g. value function
    """
    env, policy, extra = None, None, dict()

    if args is None:
        # Fall back to default arguments. By passing [], we ignore the command line arguments.
        args = get_argparser().parse_args([])

    # Hyper-parameters
    extra["hparams"] = load_hyperparameters(ex_dir)

    # Algorithm, environment, policy, and more
    algo = pyrado.load("algo.pkl", ex_dir)
    env, policy, extra = Algorithm.load_snapshot(args)

    # Check if the return types are correct. They can be None, too.
    if env is not None and not isinstance(env, (Env, EnvWrapper)):
        raise pyrado.TypeErr(given=env, expected_type=[Env, EnvWrapper])
    if policy is not None and not isinstance(policy, Policy):
        raise pyrado.TypeErr(given=policy, expected_type=Policy)
    if extra is not None and not isinstance(extra, dict):
        raise pyrado.TypeErr(given=extra, expected_type=dict)

    return env, policy, extra


def fcn_from_str(name: str) -> Callable:
    """
    Get the matching function. This method is a workaround / utility tool to intended to work with optuna. Since we can
    not pass functions directly, we pass a sting.

    :param name: name of the function
    :return: the function
    """
    if name == "to_tanh":
        return to.tanh
    elif name == "to_relu":
        return to.relu
    elif name == "to_sigmoid":
        return to.sigmoid
    elif name == "pd_linear":
        return pd_linear
    elif name == "pd_cubic":
        return pd_cubic
    elif name == "pd_capacity_21":
        return pd_capacity_21
    elif name == "pd_capacity_21_abs":
        return pd_capacity_21_abs
    elif name == "pd_capacity_32":
        return pd_capacity_32
    elif name == "pd_capacity_32_abs":
        return pd_capacity_32_abs
    else:
        raise pyrado.ValueErr(given=name, eq_constraint="'to_tanh', 'to_relu'")


def read_csv_w_replace(path: str) -> pd.DataFrame:
    """
    Custom function to read a CSV file. Turns white paces into underscores for accessing the columns as exposed
    properties, i.e. `df.prop_abc` instead of `df['prop abc']`.

    :param path: path to the CSV file
    :return: Pandas `DataFrame` with replaced chars in columns
    """
    df = pd.read_csv(path, index_col="iteration")
    # Replace whitespaces in column names
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df.columns = [c.replace("-", "_") for c in df.columns]
    df.columns = [c.replace("(", "_") for c in df.columns]
    df.columns = [c.replace(")", "_") for c in df.columns]
    return df


def load_rollouts_from_dir(
    ex_dir: str, key: Optional[str] = "rollout", file_exts: Tuple[str] = ("pt", "pkl")
) -> Tuple[List[StepSequence], List[str]]:
    """
    Crawl through the given directory, sort the files, and load all rollouts, i.e. all files that include the key.

    :param ex_dir: directory, e.g. and experiment folder
    :param key: word or part of a word that needs to the in the name of a file for it to be loaded
    :param file_exts: file extensions to be considered for loading
    :return: list of loaded rollouts, and list of file names without extension
    """
    if not osp.isdir(ex_dir):
        raise pyrado.PathErr(given=ex_dir)
    if not isinstance(key, str):
        raise pyrado.TypeErr(given=key, expected_type=str)
    if not is_iterable(file_exts):
        raise pyrado.TypeErr(given=file_exts, expected_type=Iterable)

    rollouts = []
    names = []
    for root, dirs, files in os.walk(ex_dir):
        dirs.clear()  # prevents walk() from going into subdirectories
        natural_sort(files)
        for f in files:
            f_ext = f[f.rfind(".") + 1 :]
            if key in f and f_ext in file_exts:
                name = f[: f.rfind(".")]
                names.append(name)
                rollouts.append(pyrado.load(f"{name}.{f_ext}", load_dir=root))

    if not rollouts:
        raise pyrado.ValueErr(msg="No rollouts have been found!")

    if isinstance(rollouts[0], list):
        if not check_all_types_equal(rollouts):
            raise pyrado.TypeErr(msg="Some rollout savings contain lists of rollouts, others don't!")
        # The rollout files contain lists of rollouts, flatten them
        rollouts = list(itertools.chain(*rollouts))

    return rollouts, names


def cpp_export(
    save_dir: pyrado.PathLike,
    policy: Policy,
    env: Optional[SimEnv] = None,
    policy_export_name: str = "policy_export",
    write_policy_node: bool = True,
    policy_node_name: str = "policy",
):
    """
    Convenience function to export the policy using PyTorch's scripting or tracing, and the experiment's XML
    configuration if the environment from RcsPySim.

    :param save_dir: directory to save in
    :param policy: (trained) policy
    :param env: environment the policy was trained in
    :param policy_export_name: name of the exported policy file without the file type ending
    :param write_policy_node: if `True`, write the PyTorch-based control policy into the experiment's XML configuration.
                              This requires the experiment's XML configuration to be exported beforehand.
    :param policy_node_name: name of the control policies node in the XML file, e.g. 'policy' or 'preStrikePolicy'
    """
    from pyrado.environments.rcspysim.base import RcsSim

    if not osp.isdir(save_dir):
        raise pyrado.PathErr(given=save_dir)
    if not isinstance(policy, Policy):
        raise pyrado.TypeErr(given=policy, expected_type=Policy)
    if not isinstance(policy_export_name, str):
        raise pyrado.TypeErr(given=policy_export_name, expected_type=str)

    # Use torch.jit.trace / torch.jit.script (the latter if recurrent) to generate a torch.jit.ScriptModule
    ts_module = policy.double().script()  # can be evaluated like a regular PyTorch module

    # Serialize the script module to a file and save it in the same directory we loaded the policy from
    policy_export_file = osp.join(save_dir, f"{policy_export_name}.pt")
    ts_module.save(policy_export_file)  # former .zip, and before that .pth
    print_cbt(f"Exported the loaded policy to {policy_export_file}", "g", bright=True)

    # Export the experiment config for C++
    exp_export_file = osp.join(save_dir, "ex_config_export.xml")
    if env is not None and isinstance(inner_env(env), RcsSim):
        inner_env(env).save_config_xml(exp_export_file)
        print_cbt(f"Exported experiment configuration to {exp_export_file}", "g", bright=True)

    # Open the XML file again to add the policy node
    if write_policy_node and osp.isfile(exp_export_file):
        tree = et.parse(exp_export_file)
        root = tree.getroot()
        policy_node = et.Element(policy_node_name)
        policy_node.set("type", "torch")
        policy_node.set("file", f"{policy_export_name}.pt")
        root.append(policy_node)
        tree.write(exp_export_file)
        print_cbt(f"Added {policy_export_name}.pt to the experiment configuration.", "g")
