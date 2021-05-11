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
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import pandas as pd
import torch as to

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import (
    DomainRandWrapperBuffer,
    DomainRandWrapperLive,
    remove_all_dr_wrappers,
)
from pyrado.environment_wrappers.utils import typed_env
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


def load_experiment(
    ex_dir: str, args: Any = None
) -> Tuple[Optional[Union[SimEnv, EnvWrapper]], Optional[Policy], Optional[dict]]:
    """
    Load the (training) environment and the policy.
    This helper function first tries to read the hyper-parameters yaml-file in the experiment's directory to infer
    why entities should be loaded. If no file was found, we fall back to some heuristic and hope for the best.

    :param ex_dir: experiment's parent directory
    :param args: arguments from the argument parser, pass `None` to fall back to the values from the default argparser
    :return: environment, policy, and optional output (e.g. valuefcn)
    """
    env, policy, extra = None, None, dict()

    if args is None:
        # Fall back to default arguments. By passing [], we ignore the command line arguments
        args = get_argparser().parse_args([])

    # Hyper-parameters
    extra["hparams"] = load_hyperparameters(ex_dir)

    # Algorithm specific
    algo = Algorithm.load_snapshot(load_dir=ex_dir, load_name="algo")

    if algo.name == "spota":
        # Environment
        env = pyrado.load("env.pkl", ex_dir)
        if getattr(env, "randomizer", None) is not None:
            if not isinstance(env, DomainRandWrapperBuffer):
                raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapperBuffer)
            typed_env(env, DomainRandWrapperBuffer).fill_buffer(10)
            print_cbt(f"Loaded the domain randomizer\n{env.randomizer}\nand filled it with 10 random instances.", "w")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "r")
        # Policy
        policy = pyrado.load(algo.subroutine_cand.policy, f"{args.policy_name}.pt", ex_dir, verbose=True)
        # Extra (value function)
        if isinstance(algo.subroutine_cand, ActorCritic):
            extra["vfcn"] = pyrado.load(algo.subroutine_cand.critic.vfcn, f"{args.vfcn_name}.pt", ex_dir, verbose=True)

    elif algo.name == "bayrn":
        # Environment
        env = pyrado.load("env_sim.pkl", ex_dir)
        if hasattr(env, "randomizer"):
            last_cand = to.load(osp.join(ex_dir, "candidates.pt"))[-1, :]
            env.adapt_randomizer(last_cand.numpy())
            print_cbt(f"Loaded the domain randomizer\n{env.randomizer}", "w")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "r")
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)
        # Extra (value function)
        if isinstance(algo.subroutine, ActorCritic):
            extra["vfcn"] = pyrado.load(f"{args.vfcn_name}.pt", ex_dir, obj=algo.subroutine.critic.vfcn, verbose=True)

    elif algo.name == "simopt":
        # Environment
        env = pyrado.load("env_sim.pkl", ex_dir)
        if getattr(env, "randomizer", None) is not None:
            last_cand = to.load(osp.join(ex_dir, "candidates.pt"))[-1, :]
            env.adapt_randomizer(last_cand.numpy())
            print_cbt(f"Loaded the domain randomizer\n{env.randomizer}", "w")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "r")
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.subroutine_policy.policy, verbose=True)
        # Extra (domain parameter distribution policy)
        extra["ddp_policy"] = pyrado.load("ddp_policy.pt", ex_dir, obj=algo.subroutine_distr.policy, verbose=True)

    elif algo.name in ["epopt", "udr"]:
        # Environment
        env = pyrado.load("env_sim.pkl", ex_dir)
        if getattr(env, "randomizer", None) is not None:
            if not isinstance(env, DomainRandWrapperLive):
                raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapperLive)
            print_cbt(f"Loaded the domain randomizer\n{env.randomizer}", "w")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "y")
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)
        # Extra (value function)
        if isinstance(algo.subroutine, ActorCritic):
            extra["vfcn"] = pyrado.load(f"{args.vfcn_name}.pt", ex_dir, obj=algo.subroutine.critic.vfcn, verbose=True)

    elif algo.name in ["bayessim", "npdr"]:
        # Environment
        env = pyrado.load("env_sim.pkl", ex_dir)
        if getattr(env, "randomizer", None) is not None:
            if not isinstance(env, DomainRandWrapperBuffer):
                raise pyrado.TypeErr(given=env, expected_type=DomainRandWrapperBuffer)
            typed_env(env, DomainRandWrapperBuffer).fill_buffer(10)
            print_cbt(f"Loaded the domain randomizer\n{env.randomizer}\nand filled it with 10 random instances.", "w")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "y")
            env = remove_all_dr_wrappers(env, verbose=True)
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)
        # Extra (prior, posterior, data)
        extra["prior"] = pyrado.load("prior.pt", ex_dir, verbose=True)
        # By default load the latest posterior (latest iteration and the last round)
        extra["posterior"] = algo.load_posterior(ex_dir, args.iter, args.round, obj=None, verbose=True)
        # Load the complete data or the data of the given iteration
        prefix = "" if args.iter == -1 else f"iter_{args.iter}"
        extra["data_real"] = pyrado.load(f"data_real.pt", ex_dir, prefix=prefix, verbose=True)

    elif algo.name in ["a2c", "ppo", "ppo2"]:
        # Environment
        env = pyrado.load("env.pkl", ex_dir)
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)
        # Extra (value function)
        extra["vfcn"] = pyrado.load(f"{args.vfcn_name}.pt", ex_dir, obj=algo.critic.vfcn, verbose=True)

    elif algo.name in ["hc", "pepg", "power", "cem", "reps", "nes"]:
        # Environment
        env = pyrado.load("env.pkl", ex_dir)
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)

    elif algo.name in ["dql", "sac"]:
        # Environment
        env = pyrado.load("env.pkl", ex_dir)
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)
        # Target value functions
        if algo.name == "dql":
            extra["qfcn_target"] = pyrado.load("qfcn_target.pt", ex_dir, obj=algo.qfcn_targ, verbose=True)
        elif algo.name == "sac":
            extra["qfcn_target1"] = pyrado.load("qfcn_target1.pt", ex_dir, obj=algo.qfcn_targ_1, verbose=True)
            extra["qfcn_target2"] = pyrado.load("qfcn_target2.pt", ex_dir, obj=algo.qfcn_targ_2, verbose=True)
        else:
            raise NotImplementedError

    elif algo.name == "svpg":
        # Environment
        env = pyrado.load("env.pkl", ex_dir)
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)
        # Extra (particles)
        for idx, p in enumerate(algo.particles):
            extra[f"particle{idx}"] = pyrado.load(f"particle_{idx}.pt", ex_dir, obj=algo.particles[idx], verbose=True)

    elif algo.name == "tspred":
        # Dataset
        extra["dataset"] = to.load(osp.join(ex_dir, "dataset.pt"))
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)

    elif algo.name == "sprl":
        # Environment
        env = pyrado.load("env.pkl", ex_dir)
        print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')}.", "g")
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (value function)
        if isinstance(algo._subroutine, ActorCritic):
            extra["vfcn"] = pyrado.load(f"{args.vfcn_name}.pt", ex_dir, obj=algo._subroutine.critic.vfcn, verbose=True)

    elif algo.name == "pddr":
        # Environment
        env = pyrado.load("env.pkl", ex_dir)
        # Policy
        policy = pyrado.load(f"{args.policy_name}.pt", ex_dir, obj=algo.policy, verbose=True)
        # Teachers
        extra["teacher_policies"] = algo.teacher_policies
        extra["teacher_envs"] = algo.teacher_envs
        extra["teacher_expl_strats"] = algo.teacher_expl_strats
        extra["teacher_critics"] = algo.teacher_critics
        extra["teacher_ex_dirs"] = algo.teacher_ex_dirs

    else:
        raise pyrado.TypeErr(msg="No matching algorithm name found during loading the experiment!")

    # Check if the return types are correct. They can be None, too.
    if env is not None and not isinstance(env, (SimEnv, EnvWrapper)):
        raise pyrado.TypeErr(given=env, expected_type=[SimEnv, EnvWrapper])
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
    Crawl through the given directory and load all rollouts, i.e. all files that include the key.

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
