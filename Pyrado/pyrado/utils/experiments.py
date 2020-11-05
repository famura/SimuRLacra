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

import os.path as osp
import pandas as pd
import torch as to
from typing import Callable, Any, Union

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.episodic.parameter_exploring import ParameterExploring
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.algorithms.meta.epopt import EPOpt
from pyrado.algorithms.meta.simopt import SimOpt
from pyrado.algorithms.meta.spota import SPOTA
from pyrado.algorithms.meta.udr import UDR
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.algorithms.step_based.dql import DQL
from pyrado.algorithms.step_based.sac import SAC
from pyrado.algorithms.step_based.svpg import SVPG
from pyrado.algorithms.timeseries_prediction import TSPred
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer, DomainRandWrapperLive
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper, ObsRunningNormWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import load_dict_from_yaml
from pyrado.policies.recurrent.adn import pd_linear, pd_cubic, pd_capacity_21_abs, pd_capacity_21, pd_capacity_32, \
    pd_capacity_32_abs
from pyrado.policies.base import Policy
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt
from pyrado.utils.saving_loading import load_prefix_suffix


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
    hparams_file_name = 'hyperparams.yaml'
    try:
        hparams = load_dict_from_yaml(osp.join(ex_dir, hparams_file_name))
        extra['hparams'] = hparams
    except (pyrado.PathErr, FileNotFoundError, KeyError):
        print_cbt(f'Did not find {hparams_file_name} in {ex_dir} or could not crawl the loaded hyper-parameters.',
                  'y', bright=True)

    # Algorithm specific
    algo = Algorithm.load_snapshot(load_dir=ex_dir, load_name='algo')
    if isinstance(algo, BayRn):
        # Environment
        env = load_prefix_suffix(None, 'env_sim', 'pkl', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, 'env_sim.pkl')}.", 'g')
        if hasattr(env, 'randomizer'):
            last_cand = to.load(osp.join(ex_dir, 'candidates.pt'))[-1, :]
            env.adapt_randomizer(last_cand.numpy())
            print_cbt(f'Loaded the domain randomizer\n{env.randomizer}', 'w')
        else:
            print_cbt('Loaded environment has no randomizer.', 'r')
        # Policy
        policy = load_prefix_suffix(algo.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')
        # Extra (value function)
        if isinstance(algo.subroutine, ActorCritic):
            extra['value_fcn'] = load_prefix_suffix(
                algo.subroutine.critic.value_fcn, f'{args.valuefcn_name}', 'pt', ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, f'{args.valuefcn_name}.pt')}", 'g')

    elif isinstance(algo, SPOTA):
        # Environment
        env = load_prefix_suffix(None, 'env', 'pkl', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')}.", 'g')
        if hasattr(env, 'randomizer'):
            if not isinstance(env.randomizer, DomainRandWrapperBuffer):
                raise pyrado.TypeErr(given=env.randomizer, expected_type=DomainRandWrapperBuffer)
            typed_env(env, DomainRandWrapperBuffer).fill_buffer(100)
            print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')} and filled it with 100 random instances.", 'g')
        else:
            print_cbt('Loaded environment has no randomizer.', 'r')
        # Policy
        policy = load_prefix_suffix(algo.subroutine_cand.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')
        # Extra (value function)
        if isinstance(algo.subroutine_cand, ActorCritic):
            extra['value_fcn'] = load_prefix_suffix(
                algo.subroutine_cand.critic.value_fcn, f'{args.valuefcn_name}', 'pt', ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, f'{args.valuefcn_name}.pt')}", 'g')

    elif isinstance(algo, SimOpt):
        # Environment
        env = load_prefix_suffix(None, 'env_sim', 'pkl', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, 'env_sim.pkl')}.", 'g')
        if hasattr(env, 'randomizer'):
            last_cand = to.load(osp.join(ex_dir, 'candidates.pt'))[-1, :]
            env.adapt_randomizer(last_cand.numpy())
            print_cbt(f'Loaded the domain randomizer\n{env.randomizer}', 'w')
        else:
            print_cbt('Loaded environment has no randomizer.', 'r')
        # Policy
        policy = load_prefix_suffix(algo.subroutine_policy.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')
        # Extra (domain parameter distribution policy)
        extra['ddp_policy'] = load_prefix_suffix(algo.subroutine_distr.policy, 'ddp_policy', 'pt', ex_dir, None)

    elif isinstance(algo, (EPOpt, UDR)):
        # Environment
        env = load_prefix_suffix(None, 'env_sim', 'pkl', ex_dir, None)
        if hasattr(env, 'randomizer'):
            if not isinstance(env.randomizer, DomainRandWrapperLive):
                raise pyrado.TypeErr(given=env.randomizer, expected_type=DomainRandWrapperLive)
            print_cbt(f"Loaded {osp.join(ex_dir, 'env.pkl')} with DomainRandWrapperLive randomizer.", 'g')
        else:
            print_cbt('Loaded environment has no randomizer.', 'y')
        # Policy
        policy = load_prefix_suffix(algo.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')
        # Extra (value function)
        if isinstance(algo.subroutine, ActorCritic):
            extra['value_fcn'] = load_prefix_suffix(
                algo.subroutine.critic.value_fcn, f'{args.valuefcn_name}', 'pt', ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, f'{args.valuefcn_name}.pt')}", 'g')

    elif isinstance(algo, ActorCritic):
        # Environment
        env = load_prefix_suffix(None, 'env', 'pkl', ex_dir, None)
        # Policy
        policy = load_prefix_suffix(algo.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')
        # Extra (value function)
        extra['value_fcn'] = load_prefix_suffix(algo.critic.value_fcn, f'{args.valuefcn_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.valuefcn_name}.pt')}", 'g')

    elif isinstance(algo, ParameterExploring):
        # Environment
        env = load_prefix_suffix(None, 'env', 'pkl', ex_dir, None)
        # Policy
        policy = load_prefix_suffix(algo.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')

    elif isinstance(algo, (DQL, SAC)):
        # Environment
        env = load_prefix_suffix(None, 'env', 'pkl', ex_dir, None)
        # Policy
        policy = load_prefix_suffix(algo.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')
        # Target value functions
        if isinstance(algo, DQL):
            extra['target'] = load_prefix_suffix(algo.q_targ, 'target', 'pt', ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, 'target.pt')}", 'g')
        else:
            extra['target1'] = load_prefix_suffix(algo.q_targ_1, 'target1', 'pt', ex_dir, None)
            extra['target1'] = load_prefix_suffix(algo.q_targ_2, 'target1', 'pt', ex_dir, None)
            print_cbt(f"Loaded {osp.join(ex_dir, 'target1.pt')} and {osp.join(ex_dir, 'target2.pt')}", 'g')

    elif isinstance(algo, SVPG):
        # Environment
        env = load_prefix_suffix(None, 'env', 'pkl', ex_dir, None)
        # Policy
        policy = load_prefix_suffix(algo.policy, f'{args.policy_name}', 'pt', ex_dir, None)
        print_cbt(f"Loaded {osp.join(ex_dir, f'{args.policy_name}.pt')}", 'g')
        # Extra (particles)
        for idx, p in enumerate(algo.particles):
            extra[f'particle{idx}'] = load_prefix_suffix(algo.particles[idx], f'particle_{idx}', 'pt', ex_dir, None)

    elif isinstance(algo, TSPred):
        # Dataset
        extra['dataset'] = to.load(osp.join(ex_dir, 'dataset.pt'))
        # Policy
        policy = load_prefix_suffix(algo.policy, f'{args.policy_name}', 'pt', ex_dir, None)

    else:
        raise pyrado.TypeErr(msg='No matching algorithm name found during loading the experiment!')

    # Check if the return types are correct. They can be None, too.
    if env is not None and not isinstance(env, (SimEnv, EnvWrapper)):
        raise pyrado.TypeErr(given=env, expected_type=[SimEnv, EnvWrapper])
    if policy is not None and not isinstance(policy, Policy):
        raise pyrado.TypeErr(given=policy, expected_type=Policy)
    if extra is not None and not isinstance(extra, dict):
        raise pyrado.TypeErr(given=extra, expected_type=dict)

    return env, policy, extra


def wrap_like_other_env(env_targ: Union[SimEnv, RealEnv], env_src: [SimEnv, EnvWrapper], use_downsampling: bool = False
                        ) -> Union[SimEnv, RealEnv]:
    """
    Wrap a given real environment like it's simulated counterpart (except the domain randomization of course).

    :param env_targ: target environment e.g. environment representing the physical device
    :param env_src: source environment e.g. simulation environment used for training
    :param use_downsampling: apply a wrapper that downsamples the actions if the sampling frequencies don't match
    :return: target environment
    """
    if use_downsampling and env_src.dt > env_targ.dt:
        if typed_env(env_targ, DownsamplingWrapper) is None:
            ds_factor = int(env_src.dt/env_targ.dt)
            env_targ = DownsamplingWrapper(env_targ, ds_factor)
            print_cbt(f'Wrapped the target environment with a DownsamplingWrapper of factor {ds_factor}.', 'y')
        else:
            print_cbt('The target environment was already wrapped with a DownsamplingWrapper.', 'y')

    if typed_env(env_src, ActNormWrapper) is not None:
        if typed_env(env_targ, ActNormWrapper) is None:
            env_targ = ActNormWrapper(env_targ)
            print_cbt('Wrapped the target environment with an ActNormWrapper.', 'y')
        else:
            print_cbt('The target environment was already wrapped with an ActNormWrapper.', 'y')

    if typed_env(env_src, ObsNormWrapper) is not None:
        if typed_env(env_targ, ObsNormWrapper) is None:
            env_targ = ObsNormWrapper(env_targ)
            print_cbt('Wrapped the target environment with an ObsNormWrapper.', 'y')
        else:
            print_cbt('The target environment was already wrapped with an ObsNormWrapper.', 'y')

    elif typed_env(env_src, ObsRunningNormWrapper) is not None:
        if typed_env(env_targ, ObsRunningNormWrapper) is None:
            env_targ = ObsRunningNormWrapper(env_targ)
            print_cbt('Wrapped the target environment with an ObsRunningNormWrapper.', 'y')
        else:
            print_cbt('The target environment was already wrapped with an ObsRunningNormWrapper.', 'y')

    if typed_env(env_src, ObsPartialWrapper) is not None:
        if typed_env(env_targ, ObsPartialWrapper) is None:
            env_targ = ObsPartialWrapper(
                env_targ, mask=typed_env(env_src, ObsPartialWrapper).keep_mask, keep_selected=True)
            print_cbt('Wrapped the target environment with an ObsPartialWrapper.', 'y')
        else:
            print_cbt('The target environment was already wrapped with an ObsPartialWrapper.', 'y')

    return env_targ


def fcn_from_str(name: str) -> Callable:
    """
    Get the matching function. This method is a workaround / utility tool to intended to work with optuna. Since we can
    not pass functions directly, we pass a sting.

    :param name: name of the function
    :return: the function
    """
    if name == 'to_tanh':
        return to.tanh
    elif name == 'to_relu':
        return to.relu
    elif name == 'to_sigmoid':
        return to.sigmoid
    elif name == 'pd_linear':
        return pd_linear
    elif name == 'pd_cubic':
        return pd_cubic
    elif name == 'pd_capacity_21':
        return pd_capacity_21
    elif name == 'pd_capacity_21_abs':
        return pd_capacity_21_abs
    elif name == 'pd_capacity_32':
        return pd_capacity_32
    elif name == 'pd_capacity_32_abs':
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
    df = pd.read_csv(path, index_col='iteration')
    # Replace whitespaces in column names
    df.columns = [c.replace(' ', '_') for c in df.columns]
    df.columns = [c.replace('-', '_') for c in df.columns]
    df.columns = [c.replace('(', '_') for c in df.columns]
    df.columns = [c.replace(')', '_') for c in df.columns]
    return df
