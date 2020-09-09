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
import joblib
import pandas as pd
import torch as to
from typing import Callable, Any, Optional

import pyrado
from pyrado.algorithms.a2c import A2C
from pyrado.algorithms.bayrn import BayRn
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.epopt import EPOpt
from pyrado.algorithms.hc import HC
from pyrado.algorithms.nes import NES
from pyrado.algorithms.pepg import PEPG
from pyrado.algorithms.power import PoWER
from pyrado.algorithms.ppo import PPO, PPO2
from pyrado.algorithms.reps import REPS
from pyrado.algorithms.sac import SAC
from pyrado.algorithms.spota import SPOTA
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.base import EnvWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper, ObsRunningNormWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger.experiment import load_dict_from_yaml
from pyrado.policies.adn import pd_linear, pd_cubic, pd_capacity_21_abs, pd_capacity_21, pd_capacity_32, \
    pd_capacity_32_abs
from pyrado.policies.base import Policy
from pyrado.utils.input_output import print_cbt


def load_experiment(ex_dir: str, args: Any = None) -> ([SimEnv, EnvWrapper], Policy, Optional[dict]):
    """
    Load the (training) environment and the policy.
    This helper function first tries to read the hyper-parameters yaml-file in the experiment's directory to infer
    why entities should be loaded. If no file was found, we fall back to some heuristic and hope for the best.

    :param ex_dir: experiment's parent directory
    :param args: arguments from the argument parser
    :return: environment, policy, and optional output (e.g. valuefcn)
    """
    hparams_file_name = 'hyperparams.yaml'
    env, policy, kwout = None, None, dict()

    try:
        hparams = load_dict_from_yaml(osp.join(ex_dir, hparams_file_name))
        kwout['hparams'] = hparams

        # Check which algorithm has been used for training, i.e. what can be loaded, by crawing the hyper-parameters
        # First check meta algorithms so they don't get masked by their subroutines
        if SPOTA.name in hparams.get('algo_name', ''):
            # Environment
            env = joblib.load(osp.join(ex_dir, 'init_env.pkl'))
            typed_env(env, DomainRandWrapperBuffer).fill_buffer(100)
            print_cbt(f"Loaded {osp.join(ex_dir, 'init_env.pkl')} and filled it with 100 random instances.", 'g')
            # Policy
            if args.iter == -1:
                policy = to.load(osp.join(ex_dir, 'final_policy_cand.pt'))
                print_cbt(f"Loaded {osp.join(ex_dir, 'final_policy_cand.pt')}", 'g')
            else:
                policy = to.load(osp.join(ex_dir, f'iter_{args.iter}_policy_cand.pt'))
                print_cbt(f"Loaded {osp.join(ex_dir, f'iter_{args.iter}_policy_cand.pt')}", 'g')
            # Value function (optional)
            if any([a.name in hparams.get('subrtn_name', '') for a in [PPO, PPO2, A2C]]):
                try:
                    kwout['valuefcn'] = to.load(osp.join(ex_dir, 'final_valuefcn.pt'))
                    print_cbt(f"Loaded {osp.join(ex_dir, 'final_valuefcn.pt')}", 'g')
                except FileNotFoundError:
                    kwout['valuefcn'] = to.load(osp.join(ex_dir, 'valuefcn.pt'))
                    print_cbt(f"Loaded {osp.join(ex_dir, 'valuefcn.pt')}", 'g')

        elif BayRn.name in hparams.get('algo_name', ''):
            # Environment
            env = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'env_sim.pkl')}.", 'g')
            if hasattr(env, 'randomizer'):
                last_cand = to.load(osp.join(ex_dir, 'candidates.pt'))[-1, :]
                env.adapt_randomizer(last_cand.numpy())
                print_cbt(f'Loaded the domain randomizer\n{env.randomizer}', 'w')
            # Policy
            if args.iter == -1:
                policy = to.load(osp.join(ex_dir, 'policy.pt'))
                print_cbt(f"Loaded {osp.join(ex_dir, 'policy.pt')}", 'g')
            else:
                policy = to.load(osp.join(ex_dir, f'iter_{args.iter}.pt'))
                print_cbt(f"Loaded {osp.join(ex_dir, f'iter_{args.iter}.pt')}", 'g')
            # Value function (optional)
            if any([a.name in hparams.get('subrtn_name', '') for a in [PPO, PPO2, A2C]]):
                try:
                    kwout['valuefcn'] = to.load(osp.join(ex_dir, 'final_valuefcn.pt'))
                    print_cbt(f"Loaded {osp.join(ex_dir, 'final_valuefcn.pt')}", 'g')
                except FileNotFoundError:
                    kwout['valuefcn'] = to.load(osp.join(ex_dir, 'valuefcn.pt'))
                    print_cbt(f"Loaded {osp.join(ex_dir, 'valuefcn.pt')}", 'g')

        elif EPOpt.name in hparams.get('algo_name', ''):
            # Environment
            env = joblib.load(osp.join(ex_dir, 'env.pkl'))
            # Policy
            policy = to.load(osp.join(ex_dir, 'policy.pt'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'policy.pt')}", 'g')

        elif any([a.name in hparams.get('algo_name', '') for a in [PPO, PPO2, A2C]]):
            # Environment
            env = joblib.load(osp.join(ex_dir, 'env.pkl'))
            # Policy
            policy = to.load(osp.join(ex_dir, 'policy.pt'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'policy.pt')}", 'g')
            # Value function
            kwout['valuefcn'] = to.load(osp.join(ex_dir, 'valuefcn.pt'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'valuefcn.pt')}", 'g')

        elif SAC.name in hparams.get('algo_name', ''):
            # Environment
            env = joblib.load(osp.join(ex_dir, 'env.pkl'))
            # Policy
            policy = to.load(osp.join(ex_dir, 'policy.pt'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'policy.pt')}", 'g')
            # Target value functions
            kwout['target1'] = to.load(osp.join(ex_dir, 'target1.pt'))
            kwout['target2'] = to.load(osp.join(ex_dir, 'target2.pt'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'target1.pt')} and {osp.join(ex_dir, 'target2.pt')}", 'g')

        elif any([a.name in hparams.get('algo_name', '') for a in [HC, PEPG, NES, REPS, PoWER, CEM]]):
            # Environment
            env = joblib.load(osp.join(ex_dir, 'env.pkl'))
            # Policy
            policy = to.load(osp.join(ex_dir, 'policy.pt'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'policy.pt')}", 'g')

        else:
            raise KeyError('No matching algorithm name found during loading the experiment.'
                           'Check for the algo_name field in the yaml-file.')

    except (FileNotFoundError, KeyError):
        print_cbt(f'Did not find {hparams_file_name} in {ex_dir} or could not crawl the loaded hyper-parameters.',
                  'y', bright=True)

        try:
            # Results of a standard algorithm
            env = joblib.load(osp.join(ex_dir, 'env.pkl'))
            policy = to.load(osp.join(ex_dir, 'policy.pt'))
            print_cbt(f"Loaded {osp.join(ex_dir, 'policy.pt')}", 'g')
        except FileNotFoundError:
            try:
                # Results of SPOTA
                env = joblib.load(osp.join(ex_dir, 'init_env.pkl'))
                typed_env(env, DomainRandWrapperBuffer).fill_buffer(100)
                print_cbt(f"Loaded {osp.join(ex_dir, 'init_env.pkl')} and filled it with 100 random instances.", 'g')
            except FileNotFoundError:
                # Results of BayRn
                env = joblib.load(osp.join(ex_dir, 'env_sim.pkl'))

            try:
                # Results of SPOTA
                if args.iter == -1:
                    policy = to.load(osp.join(ex_dir, 'final_policy_cand.pt'))
                    print_cbt(f'Loaded final_policy_cand.pt', 'g')
                else:
                    policy = to.load(osp.join(ex_dir, f'iter_{args.iter}_policy_cand.pt'))
                    print_cbt(f'Loaded iter_{args.iter}_policy_cand.pt', 'g')
            except FileNotFoundError:
                # Results of BayRn
                if args.iter == -1:
                    policy = to.load(osp.join(ex_dir, 'policy.pt'))
                    print_cbt(f'Loaded policy.pt', 'g')
                else:
                    policy = to.load(osp.join(ex_dir, f'iter_{args.iter}_policy.pt'))
                    print_cbt(f'Loaded iter_{args.iter}_policy.pt', 'g')

    # Check if the return types are correct
    if not isinstance(env, (SimEnv, EnvWrapper)):
        raise pyrado.TypeErr(given=env, expected_type=[SimEnv, EnvWrapper])
    if not isinstance(policy, Policy):
        raise pyrado.TypeErr(given=policy, expected_type=Policy)

    return env, policy, kwout


def wrap_like_other_env(env_targ: Env, env_src: [SimEnv, EnvWrapper], use_downsampling: bool = False) -> Env:
    """
    Wrap a given real environment like it's simulated counterpart (except the domain randomization of course).

    :param env_targ: target environment e.g. environment representing the physical device
    :param env_src: source environment e.g. simulation environment used for training
    :param use_downsampling: apply a wrapper that downsamples the actions if the sampling frequencies don't match
    :return: target environment
    """
    if use_downsampling and env_src.dt > env_targ.dt:
        ds_factor = int(env_src.dt/env_targ.dt)
        env_targ = DownsamplingWrapper(env_targ, ds_factor)
        print_cbt(f'Wrapped the env with an DownsamplingWrapper of factor {ds_factor}.', 'y')

    if typed_env(env_src, ActNormWrapper) is not None:
        env_targ = ActNormWrapper(env_targ)
        print_cbt('Wrapped the env with an ActNormWrapper.', 'y')

    if typed_env(env_src, ObsNormWrapper) is not None:
        env_targ = ObsNormWrapper(env_targ)
        print_cbt('Wrapped the env with an ObsNormWrapper.', 'y')
    elif typed_env(env_src, ObsRunningNormWrapper) is not None:
        env_targ = ObsRunningNormWrapper(env_targ)
        print_cbt('Wrapped the env with an ObsRunningNormWrapper.', 'y')

    if typed_env(env_src, ObsPartialWrapper) is not None:
        env_targ = ObsPartialWrapper(env_targ, mask=typed_env(env_src, ObsPartialWrapper).keep_mask, keep_selected=True)
        print_cbt('Wrapped the env with an ObsPartialWrapper.', 'y')

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
