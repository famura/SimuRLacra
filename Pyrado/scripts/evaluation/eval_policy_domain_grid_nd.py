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
Script to evaluate one policy in one environment for a n-dim grid (recommended: n=2) of domain parameters.

.. note::
    The domain parameters have to be scalars, otherwise the generation of the mesh grid breaks down.
"""
import datetime
import os
import os.path as osp

import numpy as np
import pandas as pd
from prettyprinter import pprint

import pyrado
from pyrado.domain_randomization.utils import param_grid
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.utils import inner_env, typed_env
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlateSim
from pyrado.logger.experiment import ask_for_experiment, save_dicts_to_yaml
from pyrado.sampling.parallel_evaluation import eval_domain_params
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import dict_arraylike_to_float
from pyrado.utils.experiments import load_experiment
from pyrado.utils.input_output import print_cbt


def evaluate_policy(args, ex_dir):
    """ Helper function to evaluate the policy from an experiment in the associated environment. """
    env, policy, _ = load_experiment(ex_dir, args)

    # Create multi-dim evaluation grid
    param_spec = dict()
    param_spec_dim = None

    if isinstance(inner_env(env), BallOnPlateSim):
        param_spec["ball_radius"] = np.linspace(0.02, 0.08, num=2, endpoint=True)
        param_spec["ball_rolling_friction_coefficient"] = np.linspace(0.0295, 0.9, num=2, endpoint=True)

    elif isinstance(inner_env(env), QQubeSwingUpSim):
        eval_num = 200
        # Use nominal values for all other parameters.
        for param, nominal_value in env.get_nominal_domain_param().items():
            param_spec[param] = nominal_value
        # param_spec["g"] = np.linspace(5.0, 15.0, num=eval_num, endpoint=True)
        param_spec["Dp"] = np.linspace(0.0, 0.0001, num=eval_num, endpoint=True)
        param_spec["Dr"] = np.linspace(0.0, 0.0006, num=eval_num, endpoint=True)
        param_spec_dim = 2

    elif isinstance(inner_env(env), QBallBalancerSim):
        # param_spec["g"] = np.linspace(7.91, 11.91, num=11, endpoint=True)
        # param_spec["m_ball"] = np.linspace(0.003, 0.3, num=11, endpoint=True)
        # param_spec["r_ball"] = np.linspace(0.01, 0.1, num=11, endpoint=True)
        param_spec["l_plate"] = np.linspace(0.275, 0.275, num=11, endpoint=True)
        param_spec["r_arm"] = np.linspace(0.0254, 0.0254, num=11, endpoint=True)
        # param_spec["J_l"] = np.linspace(5.2822e-5*0.5, 5.2822e-5*1.5, num=11, endpoint=True)
        # param_spec["J_m"] = np.linspace(4.6063e-7*0.5, 4.6063e-7*1.5, num=11, endpoint=True)
        # param_spec["K_g"] = np.linspace(60, 80, num=11, endpoint=True)
        # param_spec["eta_g"] = np.linspace(0.6, 1.0, num=11, endpoint=True)
        # param_spec["eta_m"] = np.linspace(0.49, 0.89, num=11, endpoint=True)
        # param_spec["k_m"] = np.linspace(0.006, 0.066, num=11, endpoint=True)
        # param_spec["R_m"] = np.linspace(2.6*0.5, 2.6*1.5, num=11, endpoint=True)
        # param_spec["B_eq"] = np.linspace(0.0, 0.05, num=11, endpoint=True)
        # param_spec["c_frict"] = np.linspace(0, 0.015, num=11, endpoint=True)
        # param_spec["V_thold_x_pos"] = np.linspace(0.0, 1.0, num=11, endpoint=True)
        # param_spec["V_thold_x_neg"] = np.linspace(-1., 0.0, num=11, endpoint=True)
        # param_spec["V_thold_y_pos"] = np.linspace(0.0, 1.0, num=11, endpoint=True)
        # param_spec["V_thold_y_neg"] = np.linspace(-1.0, 0, num=11, endpoint=True)
        # param_spec["offset_th_x"] = np.linspace(-5/180*np.pi, 5/180*np.pi, num=11, endpoint=True)
        # param_spec["offset_th_y"] = np.linspace(-5/180*np.pi, 5/180*np.pi, num=11, endpoint=True)

    else:
        raise NotImplementedError

    # Always add an action delay wrapper (with 0 delay by default)
    if typed_env(env, ActDelayWrapper) is None:
        env = ActDelayWrapper(env)
    # param_spec['act_delay'] = np.linspace(0, 30, num=11, endpoint=True, dtype=int)

    add_info = "-".join(param_spec.keys())

    # Create multidimensional results grid and ensure right number of rollouts
    param_list = param_grid(param_spec)
    param_list *= args.num_rollouts_per_config

    # Fix initial state (set to None if it should not be fixed)
    init_state = np.array([0.0, 0.0, 0.0, 0.0])

    # Create sampler
    pool = SamplerPool(args.num_workers)
    if args.seed is not None:
        pool.set_seed(args.seed)
        print_cbt(f"Set the random number generators' seed to {args.seed}.", "w")
    else:
        print_cbt("No seed was set", "y")

    # Sample rollouts
    ros = eval_domain_params(pool, env, policy, param_list, init_state)

    # Compute metrics
    lod = []
    for ro in ros:
        d = dict(**ro.rollout_info["domain_param"], ret=ro.undiscounted_return(), len=ro.length)
        # Simply remove the observation noise from the domain parameters
        try:
            d.pop("obs_noise_mean")
            d.pop("obs_noise_std")
        except KeyError:
            pass
        lod.append(d)

    df = pd.DataFrame(lod)
    metrics = dict(
        avg_len=df["len"].mean(),
        avg_ret=df["ret"].mean(),
        median_ret=df["ret"].median(),
        min_ret=df["ret"].min(),
        max_ret=df["ret"].max(),
        std_ret=df["ret"].std(),
    )
    pprint(metrics, indent=4)

    # Create subfolder and save
    timestamp = datetime.datetime.now()
    add_info = timestamp.strftime(pyrado.timestamp_format) + "--" + add_info
    save_dir = osp.join(ex_dir, "eval_domain_grid", add_info)
    os.makedirs(save_dir, exist_ok=True)

    save_dicts_to_yaml(
        {"ex_dir": str(ex_dir)},
        {"varied_params": list(param_spec.keys())},
        {"num_rpp": args.num_rollouts_per_config, "seed": args.seed},
        {"metrics": dict_arraylike_to_float(metrics)},
        save_dir=save_dir,
        file_name="summary",
    )
    pyrado.save(df, f"df_sp_grid_{len(param_spec) if param_spec_dim is None else param_spec_dim}d.pkl", save_dir)


if __name__ == "__main__":
    # Parse command line arguments
    g_args = get_argparser().parse_args()

    if g_args.load_all:
        if not g_args.dir:
            raise pyrado.PathErr(msg="load_all was set but no dir was given")
        if not os.path.isdir(g_args.dir):
            raise pyrado.PathErr(given=g_args.dir)

        g_ex_dirs = [tmp[0] for tmp in os.walk(g_args.dir, followlinks=True) if "policy.pt" in tmp[2]]

    elif g_args.dir is None:
        g_ex_dirs = [ask_for_experiment(hparam_list=g_args.show_hyperparameters, max_display=50)]

    else:
        g_ex_dirs = [g_args.dir]

    print(f"Evaluating all of {g_ex_dirs}.")
    for g_ex_dir in g_ex_dirs:
        print(f"Evaluating {g_ex_dir}.")
        evaluate_policy(g_args, g_ex_dir)
