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

import optuna
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO2
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.experiments import fcn_from_str
from pyrado.utils.input_output import print_cbt


def train_and_eval(trial: optuna.Trial, study_dir: str, seed: int):
    """
    Objective function for the Optuna `Study` to maximize.

    .. note::
        Optuna expects only the `trial` argument, thus we use `functools.partial` to sneak in custom arguments.

    :param trial: Optuna Trial object for hyper-parameter optimization
    :param study_dir: the parent directory for all trials in this study
    :param seed: seed value for the random number generators, pass `None` for no seeding
    :return: objective function value
    """
    # Synchronize seeds between Optuna trials
    pyrado.set_seed(seed)

    # Environment
    env_hparams = dict(dt=1 / 100.0, max_steps=600)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Learning rate scheduler
    lrs_gamma = trial.suggest_categorical("exp_lr_scheduler_gamma", [None, 0.995, 0.999])
    if lrs_gamma is not None:
        lr_sched = lr_scheduler.ExponentialLR
        lr_sched_hparam = dict(gamma=lrs_gamma)
    else:
        lr_sched, lr_sched_hparam = None, dict()

    # Policy
    policy_hparam = dict(
        hidden_sizes=trial.suggest_categorical("hidden_sizes_policy", [(32,), (64,), (32, 32), (64, 64)]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical("hidden_nonlin_policy", ["to_tanh", "to_relu"])),
    )
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Critic
    vfcn_hparam = dict(
        hidden_sizes=trial.suggest_categorical("hidden_sizes_critic", [(32,), (64,), (32, 32), (64, 64)]),
        hidden_nonlin=fcn_from_str(trial.suggest_categorical("hidden_nonlin_critic", ["to_tanh", "to_relu"])),
    )
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        batch_size=500,
        gamma=trial.suggest_loguniform("gamma_critic", 0.98, 1.0),
        lamda=trial.suggest_uniform("lamda_critic", 0.95, 1.0),
        num_epoch=trial.suggest_int("num_epoch_critic", 1, 10),
        lr=trial.suggest_loguniform("lr_critic", 1e-5, 1e-2),
        max_grad_norm=trial.suggest_categorical("max_grad_norm_critic", [None, 1.0, 5.0]),
        lr_scheduler=lr_sched,
        lr_scheduler_hparam=lr_sched_hparam,
    )
    critic = GAE(vfcn, **critic_hparam)

    # Algorithm
    algo_hparam = dict(
        num_workers=1,  # parallelize via optuna n_jobs
        max_iter=200,
        batch_size=500,
        min_steps=trial.suggest_int("num_rollouts_algo", 5, 40) * env.max_steps,
        num_epoch=trial.suggest_int("num_epoch_algo", 1, 10),
        vfcn_coeff=trial.suggest_uniform("vfcn_coeff_algo", 0.2, 2.0),
        entropy_coeff=trial.suggest_loguniform("entropy_coeff_algo", 1e-6, 1e-2),
        eps_clip=trial.suggest_uniform("eps_clip_algo", 0.05, 0.2),
        std_init=trial.suggest_uniform("std_init_algo", 0.5, 1.0),
        lr=trial.suggest_loguniform("lr_algo", 1e-5, 1e-3),
        max_grad_norm=trial.suggest_categorical("max_grad_norm_algo", [None, 1.0, 5.0]),
        lr_scheduler=lr_sched,
        lr_scheduler_hparam=lr_sched_hparam,
    )
    csv_logger = create_csv_step_logger(osp.join(study_dir, f"trial_{trial.number}"))
    algo = PPO2(osp.join(study_dir, f"trial_{trial.number}"), env, policy, critic, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode="latest", seed=seed)

    # Evaluate
    min_rollouts = 100
    sampler = ParallelRolloutSampler(
        env, policy, num_workers=1, min_rollouts=min_rollouts
    )  # parallelize via optuna n_jobs
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros]) / min_rollouts

    return mean_ret


if __name__ == "__main__":

    # Parse command line arguments
    args = get_argparser().parse_args()

    if args.dir is None:
        study_dir = setup_experiment("hyperparams", QQubeSwingUpSim.name, f"{PPO2.name}_{FNNPolicy.name}_100Hz_actnorm")
        study_dir = osp.join(pyrado.TEMP_DIR, study_dir)
        print_cbt(f"Starting a new Optuna study.", "c", bright=True)
    else:
        study_dir = args.dir
        if not osp.isdir(study_dir):
            raise pyrado.PathErr(given=study_dir)
        print_cbt(f"Continuing an existing Optuna study.", "c", bright=True)

    name = f"{QQubeSwingUpSim.name}_{PPO2.name}_{FNNPolicy.name}_100Hz_actnorm"
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:////{osp.join(study_dir, f'{name}.db')}",
        direction="maximize",
        load_if_exists=True,
    )

    # Start optimizing
    study.optimize(functools.partial(train_and_eval, study_dir=study_dir, seed=args.seed), n_trials=100, n_jobs=16)

    # Save the best hyper-parameters
    save_dicts_to_yaml(
        study.best_params,
        dict(seed=args.seed),
        save_dir=study_dir,
        file_name="best_hyperparams",
    )
