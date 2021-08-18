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
Optimize the hyper-parameters of Proximal Policy Optimization for the Quanser Qube swing-up task.
"""
import functools
import os.path as osp

import numpy as np
import optuna
import torch as to

import pyrado
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.environment_specific import QCartPoleSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.input_output import print_cbt


def train_and_eval(trial: optuna.Trial):
    dt = 0.004

    # Set up environment and policy (swing-up works reliably if is sampling frequency is >= 400 Hz)
    env = QCartPoleSwingUpSim(
        dt=dt,
        max_steps=int(6 / dt),
        long=False,
        simple_dynamics=True,
        wild_init=False,
    )
    policy = QCartPoleSwingUpAndBalanceCtrl(env.spec)
    # policy._log_K_pd.data = to.tensor([
    #     trial.suggest_uniform("log_K_pd_1", 0.0, 300.0),
    #     trial.suggest_uniform("log_K_pd_2", 0.0, 300.0),
    #     trial.suggest_uniform("log_K_pd_3", 0.0, 300.0),
    #     trial.suggest_uniform("log_K_pd_4", 0.0, 300.0),
    # ])
    policy._log_k_e.data = to.tensor(trial.suggest_uniform("log_k_e", 0.0, 50.0)).log()
    policy._log_k_p.data = to.tensor(trial.suggest_uniform("log_k_p", 0.0, 50.0)).log()

    # Simulate
    return rollout(env, policy, eval=True, seed=0).undiscounted_return()


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    ex_dir = setup_experiment(
        "hyperparams", QCartPoleSwingUpSim.name, f"{QCartPoleSwingUpSim.name}_predefined_controller"
    )
    study_dir = osp.join(pyrado.TEMP_DIR, ex_dir)
    print_cbt(f"Starting a new Optuna study.", "c", bright=True)

    name = f"{QCartPoleSwingUpSim.name}_predefined_controller"
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:////{osp.join(study_dir, f'{name}.db')}",
        direction="maximize",
        load_if_exists=True,
    )

    # Start optimizing
    study.optimize(functools.partial(train_and_eval), n_trials=200, n_jobs=8)

    # Save the best hyper-parameters
    save_dicts_to_yaml(
        study.best_params,
        dict(seed=args.seed),
        save_dir=study_dir,
        file_name="best_hyperparams",
    )
