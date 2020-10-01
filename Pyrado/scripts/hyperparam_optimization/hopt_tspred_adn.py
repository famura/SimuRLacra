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
Optimize the hyper-parameters of the Proximal Policy Optimization algorithm for the Quanser Ball-Balancer environment.
"""
import functools
import optuna
import os.path as osp
import pandas as pd
import torch as to
import torch.nn as nn
import torch.optim as optim

import pyrado
from pyrado.algorithms.timeseries_prediction import TSPred
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.adn import ADNPolicy, pd_linear
from pyrado.spaces import BoxSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_sets import TimeSeriesDataSet
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.experiments import fcn_from_str


def train_and_eval(trial: optuna.Trial, ex_dir: str, seed: int):
    """
    Objective function for the Optuna `Study` to maximize.
    
    .. note::
        Optuna expects only the `trial` argument, thus we use `functools.partial` to sneak in custom arguments.

    :param trial: Optuna Trial object for hyper-parameter optimization
    :param ex_dir: experiment's directory, i.e. the parent directory for all trials in this study
    :param seed: seed value for the random number generators, pass `None` for no seeding
    :return: objective function value
    """
    # Synchronize seeds between Optuna trials
    pyrado.set_seed(seed)

    # Load the data
    data_set_name = 'oscillation_50Hz_initpos-0.5'
    data = pd.read_csv(osp.join(pyrado.PERMA_DIR, 'time_series', f'{data_set_name}.csv'))
    if data_set_name == 'daily_min_temperatures':
        data = to.tensor(data['Temp'].values, dtype=to.get_default_dtype()).view(-1, 1)
    elif data_set_name == 'monthly_sunspots':
        data = to.tensor(data['Sunspots'].values, dtype=to.get_default_dtype()).view(-1, 1)
    elif 'oscillation' in data_set_name:
        data = to.tensor(data['Positions'].values, dtype=to.get_default_dtype()).view(-1, 1)
    else:
        raise pyrado.ValueErr(
            given=data_set_name, eq_constraint="'daily_min_temperatures', 'monthly_sunspots', "
                                               "'oscillation_50Hz_initpos-0.5', or 'oscillation_100Hz_initpos-0.4")

    # Dataset
    data_set_hparam = dict(
        name=data_set_name,
        ratio_train=0.7,
        window_size=trial.suggest_int('dataset_window_size', 1, 100),
        standardize_data=False,
        scale_min_max_data=True
    )
    dataset = TimeSeriesDataSet(data, **data_set_hparam)

    # Policy
    infspace = BoxSpace(-pyrado.inf, pyrado.inf, shape=data.unsqueeze(1).shape[1])
    policy_hparam = dict(
        dt=0.02 if 'oscillation' in data_set_name else 1.,
        obs_layer=None,
        activation_nonlin=to.tanh,
        potentials_dyn_fcn=fcn_from_str(
            trial.suggest_categorical('policy_potentials_dyn_fcn', ['pd_linear', 'pd_cubic'])),
        tau_init=trial.suggest_uniform('policy_tau_init', 1., 10.),
        tau_learnable=True,
        kappa_init=trial.suggest_categorical('policy_kappa_init', [1e-4, 1e-2]),
        kappa_learnable=True,
        capacity_learnable=True,
        potential_init_learnable=trial.suggest_categorical('policy_potential_init_learnable', [True, False]),
        init_param_kwargs=trial.suggest_categorical('policy_init_param_kwargs', [None]),
        use_cuda=False
    )
    policy = ADNPolicy(spec=EnvSpec(act_space=infspace, obs_space=infspace), **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        windowed_mode=trial.suggest_categorical('algo_windowed_mode', [True, False]),
        max_iter=1000,
        optim_class=optim.Adam,
        optim_hparam=dict(
            lr=trial.suggest_uniform('optim_lr', 5e-4, 5e-2),
            eps=trial.suggest_uniform('optim_eps', 1e-8, 1e-5),
            weight_decay=trial.suggest_uniform('optim_weight_decay', 5e-5, 5e-3)
        ),
        loss_fcn=nn.MSELoss(),
    )
    csv_logger = create_csv_step_logger(osp.join(ex_dir, f'trial_{trial.number}'))
    algo = TSPred(ex_dir, dataset, policy, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode='latest', seed=seed)

    # Evaluate
    num_init_samples = dataset.window_size
    _, loss_trn = TSPred.evaluate(policy, dataset.data_trn_inp, dataset.data_trn_targ, windowed_mode=algo.windowed_mode,
                                  num_init_samples=num_init_samples, cascaded_predictions=False)
    _, loss_tst = TSPred.evaluate(policy, dataset.data_tst_inp, dataset.data_tst_targ, windowed_mode=algo.windowed_mode,
                                  num_init_samples=num_init_samples, cascaded_predictions=False)

    return loss_trn


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Set up experiment
    ex_dir = setup_experiment('hyperparams', TSPred.name, f'{TSPred.name}_{ADNPolicy.name}')

    # Run hyper-parameter optimization
    name = f'{ex_dir.algo_name}_{ex_dir.extra_info}'  # e.g. tspred_adn
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:////{osp.join(pyrado.TEMP_DIR, ex_dir, f'{name}.db')}",
        direction='minimize',
        load_if_exists=True
    )
    study.optimize(functools.partial(train_and_eval, ex_dir=ex_dir, seed=args.seed), n_trials=100, n_jobs=16)

    # Save the best hyper-parameters
    save_list_of_dicts_to_yaml([study.best_params, dict(seed=args.seed)], ex_dir, 'best_hyperparams')
