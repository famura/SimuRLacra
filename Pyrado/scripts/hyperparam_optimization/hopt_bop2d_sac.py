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
Optimize the hyper-parameters of Soft Actor-Critic for the Ball-on-Plate environment.
"""
import functools
import optuna
import os.path as osp
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.step_based.sac import SAC
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate2DSim
from pyrado.logger.experiment import save_list_of_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.fnn import FNNPolicy
from pyrado.policies.two_headed import TwoHeadedFNNPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.spaces import BoxSpace, ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
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

    # Environment
    env_hparams = dict(
        physicsEngine='Bullet',
        dt=1/100.,
        max_steps=500
    )
    env = BallOnPlate2DSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        shared_hidden_sizes=trial.suggest_categorical('shared_hidden_sizes_policy',
                                                      [(16, 16), (32, 32), (64, 64), (16, 16, 16), (32, 32, 32)]),
        shared_hidden_nonlin=fcn_from_str(
            trial.suggest_categorical('shared_hidden_nonlin_policy', ['to_tanh', 'to_relu'])),
    )
    policy = TwoHeadedFNNPolicy(spec=env.spec, **policy_hparam)

    # Critic
    q_fcn_hparam = dict(
        hidden_sizes=trial.suggest_categorical('hidden_sizes_critic',
                                               [(16, 16), (32, 32), (64, 64), (16, 16, 16), (32, 32, 32)]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical('hidden_nonlin_critic', ['to_tanh', 'to_relu'])),
    )
    obsact_space = BoxSpace.cat([env.obs_space, env.act_space])
    q_fcn_1 = FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **q_fcn_hparam)
    q_fcn_2 = FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **q_fcn_hparam)

    # Algorithm
    algo_hparam = dict(
        num_workers=1,  # parallelize via optuna n_jobs
        max_iter=100*env.max_steps,
        min_steps=trial.suggest_categorical('min_steps_algo', [1]),  # 10, env.max_steps, 10*env.max_steps
        memory_size=trial.suggest_loguniform('memory_size_algo', 1e2*env.max_steps, 1e4*env.max_steps),
        tau=trial.suggest_uniform('tau_algo', 0.99, 1.),
        ent_coeff_init=trial.suggest_uniform('ent_coeff_init_algo', 0.1, 0.9),
        learn_ent_coeff=trial.suggest_categorical('learn_ent_coeff_algo', [True, False]),
        standardize_rew=trial.suggest_categorical('standardize_rew_algo', [False]),
        gamma=trial.suggest_uniform('gamma_algo', 0.99, 1.),
        target_update_intvl=trial.suggest_categorical('target_update_intvl_algo', [1, 5]),
        num_batch_updates=trial.suggest_categorical('num_batch_updates_algo', [1, 5]),
        batch_size=trial.suggest_categorical('batch_size_algo', [128, 256, 512]),
        lr=trial.suggest_loguniform('lr_algo', 1e-5, 1e-3),
    )
    csv_logger = create_csv_step_logger(osp.join(ex_dir, f'trial_{trial.number}'))
    algo = SAC(ex_dir, env, policy, q_fcn_1, q_fcn_2, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode='latest', seed=seed)

    # Evaluate
    min_rollouts = 1000
    sampler = ParallelRolloutSampler(env, policy, num_workers=1,
                                     min_rollouts=min_rollouts)  # parallelize via optuna n_jobs
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros])/min_rollouts

    return mean_ret


if __name__ == '__main__':
    # Parse command line arguments
    args = get_argparser().parse_args()

    ex_dir = setup_experiment('hyperparams', BallOnPlate2DSim.name, f'{SAC.name}_{FNNPolicy.name}_100Hz',
                              seed=args.seed)

    # Run hyper-parameter optimization
    name = f'{ex_dir.algo_name}_{ex_dir.extra_info}'  # e.g. bop2d_sac_fnn_100Hz
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:////{osp.join(pyrado.TEMP_DIR, ex_dir, f'{name}.db')}",
        direction='maximize',
        pruner=MedianPruner(),
        load_if_exists=True
    )
    study.optimize(functools.partial(train_and_eval, ex_dir=ex_dir, seed=args.seed), n_trials=50, n_jobs=6)

    # Save the best hyper-parameters
    save_list_of_dicts_to_yaml([study.best_params, dict(seed=args.seed)], ex_dir, 'best_hyperparams')
