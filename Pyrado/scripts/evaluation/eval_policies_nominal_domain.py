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
Script to evaluate multiple policies in one environment using the nominal domain parameters.
"""
import os.path as osp
import pandas as pd
from prettyprinter import pprint

import pyrado
from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerSim
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim, QCartPoleStabSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.sampling.parallel_evaluation import eval_nominal_domain
from pyrado.sampling.sampler_pool import SamplerPool
from pyrado.utils.argparser import get_argparser
from pyrado.utils.checks import check_all_lengths_equal
from pyrado.utils.data_types import dict_arraylike_to_float
from pyrado.utils.experiments import load_experiment
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.utils.input_output import print_cbt


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    if args.max_steps == pyrado.inf:
        args.max_steps = 2500
        print_cbt(f"Set maximum number of time steps to {args.max_steps}", "y")

    if args.env_name == QBallBalancerSim.name:
        # Create the environment for evaluating
        env = QBallBalancerSim(dt=args.dt, max_steps=args.max_steps)

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

    elif args.env_name in [QCartPoleStabSim.name, QCartPoleSwingUpSim.name]:
        # Create the environment for evaluating
        if args.env_name == QCartPoleSwingUpSim.name:
            env = QCartPoleSwingUpSim(dt=args.dt, max_steps=args.max_steps)
        else:
            env = QCartPoleStabSim(dt=args.dt, max_steps=args.max_steps)

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
            given=args.env_name,
            eq_constraint=f"{QBallBalancerSim.name}, {QCartPoleStabSim.name}," f"or {QCartPoleSwingUpSim.name}",
        )

    if not check_all_lengths_equal([prefixes, ex_names, ex_labels]):
        raise pyrado.ShapeErr(
            msg=f"The lengths of prefixes, ex_names, and ex_labels must be equal, "
            f"but they are {len(prefixes)}, {len(ex_names)}, and {len(ex_labels)}!"
        )

    # Loading the policies
    ex_dirs = [osp.join(p, e) for p, e in zip(prefixes, ex_names)]
    env_sim_list = []
    policy_list = []
    for ex_dir in ex_dirs:
        _, policy, _ = load_experiment(ex_dir, args)
        policy_list.append(policy)

    # Fix initial state (set to None if it should not be fixed)
    init_state_list = [None] * args.num_ro_per_config

    # Crate empty data frame
    df = pd.DataFrame(columns=["policy", "ret", "len"])

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
        ros = eval_nominal_domain(pool, env, policy, init_state_list)

        # Compute results metrics
        rets = [ro.undiscounted_return() for ro in ros]
        lengths = [float(ro.length) for ro in ros]  # int values are not numeric in pandas
        df = df.append(pd.DataFrame(dict(policy=ex_labels[i], ret=rets, len=lengths)), ignore_index=True)

    metrics = dict(
        avg_len=df.groupby("policy").mean()["len"].to_dict(),
        avg_ret=df.groupby("policy").mean()["ret"].to_dict(),
        median_ret=df.groupby("policy").median()["ret"].to_dict(),
        min_ret=df.groupby("policy").min()["ret"].to_dict(),
        max_ret=df.groupby("policy").max()["ret"].to_dict(),
        std_ret=df.groupby("policy").std()["ret"].to_dict(),
    )
    pprint(metrics, indent=4)

    # Create sub-folder and save
    save_dir = setup_experiment("multiple_policies", args.env_name, "nominal", base_dir=pyrado.EVAL_DIR)

    save_dicts_to_yaml(
        {"ex_dirs": ex_dirs},
        {"num_rpp": args.num_ro_per_config, "seed": args.seed},
        {"metrics": dict_arraylike_to_float(metrics)},
        save_dir=save_dir,
        file_name="summary",
    )
    df.to_pickle(osp.join(save_dir, "df_nom_mp.pkl"))
