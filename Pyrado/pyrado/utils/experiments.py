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
import pandas as pd
import torch as to
from typing import Callable, Any, Union, List, Optional, Tuple, Iterable

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer, DomainRandWrapperLive
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import load_dict_from_yaml
from pyrado.policies.recurrent.adn import (
    pd_linear,
    pd_cubic,
    pd_capacity_21_abs,
    pd_capacity_21,
    pd_capacity_32,
    pd_capacity_32_abs,
)
from pyrado.policies.base import Policy
from pyrado.sampling.step_sequence import StepSequence
from pyrado.utils.argparser import get_argparser
from pyrado.utils.checks import check_all_types_equal, is_iterable
from pyrado.utils.input_output import print_cbt


def load_experiment(ex_dir: str, args: Any = None) -> (Union[SimEnv, EnvWrapper], Policy, dict):
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
    hparams_file_name = "hyperparams.yaml"
    try:
        hparams = load_dict_from_yaml(osp.join(ex_dir, hparams_file_name))
        extra["hparams"] = hparams
    except (pyrado.PathErr, FileNotFoundError, KeyError):
        print_cbt(
            f"Did not find {hparams_file_name} in {ex_dir} or could not crawl the loaded hyper-parameters.",
            "y",
            bright=True,
        )

    # Algorithm specific
    algo = Algorithm.load_snapshot(load_dir=ex_dir, load_name="algo")
    if algo.name == "bayrn":
        # Environment
        env = pyrado.load(None, "env_sim", "pkl", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, 'env_sim.pkl')}.", "g")
        if hasattr(env, "randomizer"):
            last_cand = to.load(osp.join(ex_dir, "candidates.pt"))[-1, :]
            env.adapt_randomizer(last_cand.numpy())
            print_cbt(f"Loaded the domain randomizer\n{env.randomizer}", "w")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "r")
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (value function)
        if isinstance(algo.subroutine, ActorCritic):
            extra["vfcn"] = pyrado.load(algo.subroutine.critic.vfcn, f"{args.vfcn_name}", "pt", ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, f'{args.vfcn_name}.pt')}", "g")

    elif algo.name == "spota":
        # Environment
        env = pyrado.load(None, "env", "pkl", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')}.", "g")
        if getattr(env, "randomizer", None) is not None:
            if not isinstance(env.randomizer, DomainRandWrapperBuffer):
                raise pyrado.TypeErr(given=env.randomizer, expected_type=DomainRandWrapperBuffer)
            typed_env(env, DomainRandWrapperBuffer).fill_buffer(100)
            print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')} and filled it with 100 random instances.", "g")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "r")
        # Policy
        policy = pyrado.load(algo.subroutine_cand.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (value function)
        if isinstance(algo.subroutine_cand, ActorCritic):
            extra["vfcn"] = pyrado.load(algo.subroutine_cand.critic.vfcn, f"{args.vfcn_name}", "pt", ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, f'{args.vfcn_name}.pt')}", "g")

    elif algo.name == "simopt":
        # Environment
        env = pyrado.load(None, "env_sim", "pkl", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, 'env_sim.pkl')}.", "g")
        if getattr(env, "randomizer", None) is not None:
            last_cand = to.load(osp.join(ex_dir, "candidates.pt"))[-1, :]
            env.adapt_randomizer(last_cand.numpy())
            print_cbt(f"Loaded the domain randomizer\n{env.randomizer}", "w")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "r")
        # Policy
        policy = pyrado.load(algo.subroutine_policy.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (domain parameter distribution policy)
        extra["ddp_policy"] = pyrado.load(algo.subroutine_distr.policy, "ddp_policy", "pt", ex_dir, None)

    elif algo.name in ["epopt", "udr"]:
        # Environment
        env = pyrado.load(None, "env_sim", "pkl", ex_dir, None)
        if getattr(env, "randomizer", None) is not None:
            if not isinstance(env.randomizer, DomainRandWrapperLive):
                raise pyrado.TypeErr(given=env.randomizer, expected_type=DomainRandWrapperLive)
            print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')} with DomainRandWrapperLive randomizer.", "g")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "y")
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (value function)
        if isinstance(algo.subroutine, ActorCritic):
            extra["vfcn"] = pyrado.load(algo.subroutine.critic.vfcn, f"{args.vfcn_name}", "pt", ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, f'{args.vfcn_name}.pt')}", "g")

    elif algo.name == "lfi":
        # Environment
        env = pyrado.load(None, "env_sim", "pkl", ex_dir, None)
        if getattr(env, "randomizer", None) is not None:
            if not isinstance(env.randomizer, DomainRandWrapperBuffer):
                raise pyrado.TypeErr(given=env.randomizer, expected_type=DomainRandWrapperBuffer)
            typed_env(env, DomainRandWrapperBuffer).fill_buffer(10)
            print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')} and filled it with 10 random instances.", "g")
        else:
            print_cbt("Loaded environment has no randomizer, or it is None.", "y")
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (prior, posterior, observations)
        extra["prior"] = pyrado.load(None, "prior", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'prior.pt')}", "g")
        if args.iter == -1:
            # Load the complete history
            extra["posterior"] = pyrado.load(None, "posterior", "pt", ex_dir, None)
            extra["observations_real"] = pyrado.load(None, "observations_real", "pt", ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, f'posterior.pt')}", "g")
            print_cbt(f"Loaded {osp.join(ex_dir, f'observations_real.pt')}", "g")
        else:
            # Load only one iteration
            extra["posterior"] = pyrado.load(
                None, "posterior", "pt", ex_dir, meta_info=dict(prefix=f"iter_{args.iter}")
            )
            extra["observations_real"] = pyrado.load(
                None, f"observations_real", "pt", ex_dir, meta_info=dict(prefix=f"iter_{args.iter}")
            )
            print_cbt(f"Loaded {osp.join(ex_dir, f'iter_{args.iter}_posterior.pt')}", "g")
            print_cbt(f"Loaded {osp.join(ex_dir, f'iter_{args.iter}_observations_real.pt')}", "g")

    elif algo.name in ["a2c", "ppo", "ppo2"]:
        # Environment
        env = pyrado.load(None, "env", "pkl", ex_dir, None)
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (value function)
        extra["vfcn"] = pyrado.load(algo.critic.vfcn, f"{args.vfcn_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.vfcn_name}.pt')}", "g")

    elif algo.name in ["hc", "pepg", "power", "cem", "reps", "nes"]:
        # Environment
        env = pyrado.load(None, "env", "pkl", ex_dir, None)
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")

    elif algo.name in ["dql", "sac"]:
        # Environment
        env = pyrado.load(None, "env", "pkl", ex_dir, None)
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Target value functions
        if algo.name == "dql":
            extra["qfcn_target"] = pyrado.load(algo.qfcn_targ, "qfcn_target", "pt", ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, 'qfcn_target.pt')}", "g")
        elif algo.name == "sac":
            extra["qfcn_target1"] = pyrado.load(algo.qfcn_targ_1, "qfcn_target1", "pt", ex_dir, None)
            extra["qfcn_target2"] = pyrado.load(algo.qfcn_targ_2, "qfcn_target2", "pt", ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, 'qfcn_target1.pt')} and {osp.join(ex_dir, 'qfcn_target2.pt')}", "g")
        else:
            raise NotImplementedError

    elif algo.name == "svpg":
        # Environment
        env = pyrado.load(None, "env", "pkl", ex_dir, None)
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", "g")
        # Extra (particles)
        for idx, p in enumerate(algo.particles):
            extra[f"particle{idx}"] = pyrado.load(algo.particles[idx], f"particle_{idx}", "pt", ex_dir, None)

    elif algo.name == "tspred":
        # Dataset
        extra["dataset"] = to.load(osp.join(ex_dir, "dataset.pt"))
        # Policy
        policy = pyrado.load(algo.policy, f"{args.policy_name}", "pt", ex_dir, None)

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
) -> List[StepSequence]:
    """
    Crawl through the given directory and load all rollouts, i.e. all files that include the key.

    :param ex_dir: directory, e.g. and experiment folder
    :param key: word or part of a word that needs to the in the name of a file for it to be loaded
    :param file_exts: file extensions to be considered for loading
    :return: list of loaded rollouts
    """
    if not osp.isdir(ex_dir):
        raise pyrado.PathErr(given=ex_dir)
    if not isinstance(key, str):
        raise pyrado.TypeErr(given=key, expected_type=str)
    if not is_iterable(file_exts):
        raise pyrado.TypeErr(given=file_exts, expected_type=Iterable)

    rollouts = []
    for root, dirs, files in os.walk(ex_dir):
        dirs.clear()  # prevents walk() from going into subdirectories
        rollouts = []
        for f in files:
            f_ext = f[f.rfind(".") + 1 :]
            if key in f and f_ext in file_exts:
                rollouts.append(pyrado.load(None, name=f[: f.rfind(".")], file_ext=f_ext, load_dir=root))

    if not rollouts:
        raise pyrado.ValueErr(msg="No rollouts have been found!")

    if isinstance(rollouts[0], list):
        if not check_all_types_equal(rollouts):
            raise pyrado.TypeErr(msg="Some rollout savings contain lists of rollouts, others don't!")
        # The rollout files contain lists of rollouts, flatten them
        rollouts = list(itertools.chain(*rollouts))

    return rollouts
