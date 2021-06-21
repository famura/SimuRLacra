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

"""
Script to evaluate multiple policies in one environment using a range (1D grid) of domain parameters
"""
import os.path as osp

import numpy as np
import pandas as pd
from prettyprinter import pprint

import pyrado
from pyrado.domain_randomization.utils import param_grid, wrap_like_other_env
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleStabSim, QCartPoleSwingUpSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import ask_for_experiment, save_dicts_to_yaml, setup_experiment
from pyrado.sampling.parallel_evaluation import eval_domain_params
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.utils.argparser import get_argparser
from pyrado.utils.checks import check_all_lengths_equal
from pyrado.utils.data_types import dict_arraylike_to_float
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if args.max_steps == pyrado.inf:
        args.max_steps = 2000
        print_cbt(f"Set maximum number of time steps to {args.max_steps}", "y")

    # Get the experiment's directory to load from
    experiment = ask_for_experiment(hparam_list=args.show_hparams)
    env, policy, _ = load_experiment(experiment, args)
    env_name = env.name
    dt = env.dt

    # Create one-dim evaluation grid
    param_spec = dict()

    if env_name == QBallBalancerSim.name:
        # Create the environment for evaluating
        env = QBallBalancerSim(dt=dt, max_steps=args.max_steps, load_experimental_tholds=True)

        # param_spec['g'] = np.linspace(8.91, 12.91, num=11, endpoint=True)
        # param_spec['m_ball'] = np.linspace(0.001, 0.033, num=11, endpoint=True)
        # param_spec['r_ball'] = np.linspace(0.01, 0.1, num=11, endpoint=True)
        # param_spec['r_arm'] = np.linspace(0.0254*0.3, 0.0254*1.7, num=11, endpoint=True)
        # param_spec['l_plate'] = np.linspace(0.275*0.3, 0.275*1.7, num=11, endpoint=True)
        # param_spec['J_l'] = np.linspace(5.2822e-5 * 0.5, 5.2822e-5 * 1.5, num=11, endpoint=True)
        # param_spec['J_m'] = np.linspace(4.6063e-7*0.5, 4.6063e-7*1.5, num=11, endpoint=True)
        # param_spec['K_g'] = np.linspace(70*0.5, 70*1.5, num=11, endpoint=True)
        # param_spec['eta_g'] = np.linspace(0.6, 1.0, num=11, endpoint=True)
        # param_spec['eta_m'] = np.linspace(0.49, 0.89, num=11, endpoint=True)
        # param_spec['k_m'] = np.linspace(0.0077*0.3, 0.0077*1.7, num=11, endpoint=True)
        # param_spec['k_m'] = np.linspace(0.004, 0.012, num=11, endpoint=True)
        # param_spec['R_m'] = np.linspace(2.6*0.5, 2.6*1.5, num=11, endpoint=True)
        # param_spec['B_eq'] = np.linspace(0.0, 0.2, num=11, endpoint=True)
        # param_spec['c_frict'] = np.linspace(0, 0.15, num=11, endpoint=True)
        # param_spec['V_thold_x_pos'] = np.linspace(0.0, 1.5, num=11, endpoint=True)
        # param_spec['V_thold_x_neg'] = np.linspace(-1.5, 0.0, num=11, endpoint=True)
        # param_spec['V_thold_y_pos'] = np.linspace(0.0, 1.5, num=11, endpoint=True)
        # param_spec['V_thold_y_neg'] = np.linspace(-1.5, 0, num=11, endpoint=True)
        # param_spec['offset_th_x'] = np.linspace(-15./180*np.pi, 15./180*np.pi, num=11, endpoint=True)
        # param_spec['offset_th_y'] = np.linspace(-15./180*np.pi, 15./180*np.pi, num=11, endpoint=True)

        # Get the experiments' directories to load from
        prefixes = [
            osp.join(pyrado.EXP_DIR, "ENV_NAME", "ALGO_NAME"),
        ]
        ex_names = [
            "",
        ]
        ex_labels = [
            "",
        ]

    elif env_name in [QCartPoleStabSim.name, QCartPoleSwingUpSim.name]:
        # Create the environment for evaluating
        if env_name == QCartPoleSwingUpSim.name:
            env = QCartPoleSwingUpSim(dt=dt, max_steps=args.max_steps)
        else:
            env = QCartPoleStabSim(dt=dt, max_steps=args.max_steps)

        # param_spec['g'] = np.linspace(9.8*10.7, 9.81*1.3, num=11 endpoint=True)
        param_spec["m_cart"] = np.linspace(0.38 * 0.7, 0.38 * 1.3, num=11, endpoint=True)
        # param_spec['l_rail'] = np.linspace(0.841*0.7, 0.841*1.3, num=11, endpoint=True)
        # param_spec['eta_m'] = np.linspace(0.9*0.7, 0.9*1.3, num=11, endpoint=True)
        # param_spec['eta_g'] = np.linspace(0.9*0.7, 0.9*1.3, num=11, endpoint=True)
        # param_spec['K_g'] = np.linspace(3.71*0.7, 3.71*1.3, num=11, endpoint=True)
        # param_spec['J_m'] = np.linspace(3.9e-7*0.7, 3.9e-7*1.3, num=11, endpoint=True)
        # param_spec['r_mp'] = np.linspace(6.35e-3*0.7, 6.35e-3*1.3, num=11, endpoint=True)
        # param_spec['R_m'] = np.linspace(2.6*0.7, 2.6*1.3, num=11, endpoint=True)
        # param_spec['k_m'] = np.linspace(7.67e-3*0.7, 7.67e-3*1.3, num=11, endpoint=True)
        # param_spec['B_pole'] = np.linspace(0.0024*0.7, 0.0024*1.3, num=11, endpoint=True)
        # param_spec['B_eq'] = np.linspace(5.4*0.7, 5.4*1.3, num=11, endpoint=True)
        # param_spec['m_pole'] = np.linspace(0.127*0.7, 0.127*1.3, num=11, endpoint=True)
        # param_spec['l_pole'] = np.linspace(0.641/2*0.7, 0.641/2*1.3, num=11, endpoint=True)

        # Get the experiments' directories to load from
        prefixes = [
            osp.join(pyrado.EXP_DIR, "ENV_NAME", "ALGO_NAME"),
        ]
        ex_names = [
            "",
        ]
        ex_labels = [
            "",
        ]

    elif env_name == QQubeSwingUpSim.name:
        env = QQubeSwingUpSim(dt=dt, max_steps=args.max_steps)

        # param_spec['g'] = np.linspace(9.81*0.7, 9.81*1.3, num=11, endpoint=True)
        # param_spec['Rm'] = np.linspace(8.4*0.7, 8.4*1.3, num=11, endpoint=True)
        # param_spec['km'] = np.linspace(0.042*0.7, 0.042*1.3, num=11, endpoint=True)
        # param_spec['mass_rot_pole'] = np.linspace(0.095*0.7, 0.095*1.3, num=11, endpoint=True)
        # param_spec['Lr'] = np.linspace(0.085*0.7, 0.085*1.3, num=11, endpoint=True)
        # param_spec['Dr'] = np.linspace(5e-6*0.2, 5e-6*5, num=11, endpoint=True)  # 5e-6
        # param_spec['Mp'] = np.linspace(0.024*0.7, 0.024*1.3, num=11, endpoint=True)
        # param_spec['Lp'] = np.linspace(0.129*0.7, 0.129*1.3, num=11, endpoint=True)
        # param_spec['Dp'] = np.linspace(1e-6*0.2, 1e-6n*5, num=11, endpoint=True)  # 1e-6

        # Get the experiments' directories to load from
        prefixes = [
            osp.join(pyrado.EXP_DIR, "ENV_NAME", "ALGO_NAME"),
        ]
        ex_names = [
            "",
        ]
        ex_labels = [
            "",
        ]

    else:
        raise pyrado.ValueErr(
            given=env_name,
            eq_constraint=f"{QBallBalancerSim.name}, {QCartPoleStabSim.name},"
            f"{QCartPoleSwingUpSim.name}, or {QQubeSwingUpSim.name}",
        )

        # Always add an action delay wrapper (with 0 delay by default)
    if typed_env(env, ActDelayWrapper) is None:
        env = ActDelayWrapper(env)
        # param_spec['act_delay'] = np.linspace(0, 60, num=21, endpoint=True, dtype=int)

    if not len(param_spec.keys()) == 1:
        raise pyrado.ValueErr(msg="Do not vary more than one domain parameter for this script! (Check action delay.)")
    varied_param_key = "".join(param_spec.keys())  # to get a str

    if not check_all_lengths_equal([prefixes, ex_names, ex_labels]):
        raise pyrado.ShapeErr(
            msg=f"The lengths of prefixes, ex_names, and ex_labels must be equal, "
            f"but they are {len(prefixes)}, {len(ex_names)}, and {len(ex_labels)}!"
        )

    if experiment and env and policy:
        # Load only the single policy if it was set by asking for the policy.
        ex_dirs = [str(experiment)]
        env_sim_list = [env]
        policy_list = [policy]
    else:
        # Loading the policies
        ex_dirs = [osp.join(p, e) for p, e in zip(prefixes, ex_names)]
        env_sim_list = []
        policy_list = []
        for ex_dir in ex_dirs:
            _, policy, _ = load_experiment(ex_dir, args)
            policy_list.append(policy)

    # Create one-dim results grid and ensure right number of rollouts
    param_list = param_grid(param_spec)
    param_list *= args.num_rollouts_per_config

    # Fix initial state (set to None if it should not be fixed)
    init_state = None

    # Crate empty data frame
    df = pd.DataFrame(columns=["policy", "ret", "len", varied_param_key])

    # Evaluate all policies
    for i, (env_sim, policy) in enumerate(zip(env_sim_list, policy_list)):
        # Create a new sampler pool for every policy to synchronize the random seeds i.e. init states
        pool = SamplerPool(args.num_workers)

        # Seed the sampler
        if args.seed is not None:
            pool.set_seed(args.seed)
            print_cbt(f"Set the random number generators' seed to {args.seed}.", "w")
        else:
            print_cbt("No seed was set", "y")

        # Add the same wrappers as during training
        env = wrap_like_other_env(env, env_sim)

        # Sample rollouts
        ros = eval_domain_params(pool, env, policy, param_list, init_state)

        # Compute results metrics
        rets = [ro.undiscounted_return() for ro in ros]
        lengths = [float(ro.length) for ro in ros]  # int values are not numeric in pandas
        vaired_param_values = [ro.rollout_info["domain_param"][varied_param_key] for ro in ros]
        varied_param = {varied_param_key: vaired_param_values}
        df = df.append(
            pd.DataFrame(dict(policy=ex_labels[i], ret=rets, len=lengths, **varied_param)), ignore_index=True
        )

    metrics = dict(
        avg_len=df.groupby("policy").mean()["len"].to_dict(),
        avg_ret=df.groupby("policy").mean()["ret"].to_dict(),
        median_ret=df.groupby("policy").median()["ret"].to_dict(),
        min_ret=df.groupby("policy").min()["ret"].to_dict(),
        max_ret=df.groupby("policy").max()["ret"].to_dict(),
        std_ret=df.groupby("policy").std()["ret"].to_dict(),
        quantile5_ret=df.groupby("policy").quantile(q=0.05)["ret"].to_dict(),
        quantile95_ret=df.groupby("policy").quantile(q=0.95)["ret"].to_dict(),
    )
    pprint(metrics, indent=4)

    # Create subfolder and save
    save_dir = setup_experiment("multiple_policies", env_name, varied_param_key, base_dir=pyrado.EVAL_DIR)

    save_dicts_to_yaml(
        {"ex_dirs": ex_dirs},
        {
            "varied_param": varied_param_key,
            "num_rpp": args.num_rollouts_per_config,
            "seed": args.seed,
            "dt": dt,
            "max_steps": args.max_steps,
        },
        {"metircs": dict_arraylike_to_float(metrics)},
        save_dir=save_dir,
        file_name="summary",
    )
    df.to_pickle(osp.join(save_dir, "df_mp_grid_1d.pkl"))
