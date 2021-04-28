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
Optimize the hyper-parameters of SimOpt for the Quanser Qube swing-up task.
"""
import functools
import os.path as osp

import optuna
import torch as to

import pyrado
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.episodic.sysid_via_episodic_rl import SysIdViaEpisodicRL
from pyrado.algorithms.meta.simopt import SimOpt
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive, MetaDomainRandWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.special.domain_distribution import DomainDistrParamPolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec
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

    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=600)
    env_real = QQubeSwingUpSim(**env_hparams)
    env_real.domain_param = dict(
        Mr=0.095 * 0.9,  # 0.095*0.9 = 0.0855
        Mp=0.024 * 1.1,  # 0.024*1.1 = 0.0264
        Lr=0.085 * 0.9,  # 0.085*0.9 = 0.0765
        Lp=0.129 * 1.1,  # 0.129*1.1 = 0.1419
    )

    env_sim = QQubeSwingUpSim(**env_hparams)
    randomizer = DomainRandomizer(
        NormalDomainParam(name="Mr", mean=0.0, std=1e6, clip_lo=1e-3),
        NormalDomainParam(name="Mp", mean=0.0, std=1e6, clip_lo=1e-3),
        NormalDomainParam(name="Lr", mean=0.0, std=1e6, clip_lo=1e-3),
        NormalDomainParam(name="Lp", mean=0.0, std=1e6, clip_lo=1e-3),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {
        0: ("Mr", "mean"),
        1: ("Mr", "std"),
        2: ("Mp", "mean"),
        3: ("Mp", "std"),
        4: ("Lr", "mean"),
        5: ("Lr", "std"),
        6: ("Lp", "mean"),
        7: ("Lp", "std"),
    }
    trafo_mask = [True] * 8
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Subroutine for policy improvement
    behav_policy_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    behav_policy = FNNPolicy(spec=env_sim.spec, **behav_policy_hparam)
    vfcn_hparam = dict(hidden_sizes=[64, 64], hidden_nonlin=to.tanh)
    vfcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.9885,
        lamda=0.9648,
        num_epoch=2,
        batch_size=500,
        standardize_adv=False,
        lr=5.792e-4,
        max_grad_norm=1.0,
    )
    critic = GAE(vfcn, **critic_hparam)
    subrtn_policy_hparam = dict(
        max_iter=200,
        min_steps=3 * 23 * env_sim.max_steps,
        num_epoch=7,
        eps_clip=0.0744,
        batch_size=500,
        std_init=0.9074,
        lr=3.446e-04,
        max_grad_norm=1.0,
        num_workers=1,
    )
    subrtn_policy = PPO(study_dir, env_sim, behav_policy, critic, **subrtn_policy_hparam)

    # Subroutine for system identification
    prior_std_denom = trial.suggest_uniform("prior_std_denom", 5, 20)
    prior = DomainRandomizer(
        NormalDomainParam(name="Mr", mean=0.095, std=0.095 / prior_std_denom),
        NormalDomainParam(name="Mp", mean=0.024, std=0.024 / prior_std_denom),
        NormalDomainParam(name="Lr", mean=0.085, std=0.085 / prior_std_denom),
        NormalDomainParam(name="Lp", mean=0.129, std=0.129 / prior_std_denom),
    )
    ddp_policy = DomainDistrParamPolicy(
        mapping=dp_map,
        trafo_mask=trafo_mask,
        prior=prior,
        scale_params=trial.suggest_categorical("ddp_policy_scale_params", [True, False]),
    )
    subsubrtn_distr_hparam = dict(
        max_iter=trial.suggest_categorical("subsubrtn_distr_max_iter", [20]),
        pop_size=trial.suggest_int("pop_size", 50, 500),
        num_init_states_per_domain=1,
        num_is_samples=trial.suggest_int("num_is_samples", 5, 20),
        expl_std_init=trial.suggest_loguniform("expl_std_init", 1e-3, 1e-1),
        expl_std_min=trial.suggest_categorical("expl_std_min", [1e-4]),
        extra_expl_std_init=trial.suggest_loguniform("expl_std_init", 1e-3, 1e-1),
        extra_expl_decay_iter=trial.suggest_int("extra_expl_decay_iter", 0, 10),
        num_workers=1,
    )
    csv_logger = create_csv_step_logger(osp.join(study_dir, f"trial_{trial.number}"))
    subsubrtn_distr = CEM(study_dir, env_sim, ddp_policy, **subsubrtn_distr_hparam, logger=csv_logger)
    obs_vel_weight = trial.suggest_loguniform("obs_vel_weight", 1, 100)
    subrtn_distr_hparam = dict(
        metric=None,
        obs_dim_weight=[1, 1, 1, 1, obs_vel_weight, obs_vel_weight],
        num_rollouts_per_distr=trial.suggest_int("num_rollouts_per_distr", 20, 100),
        num_workers=1,
    )
    subrtn_distr = SysIdViaEpisodicRL(subsubrtn_distr, behav_policy, **subrtn_distr_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=trial.suggest_categorical("algo_max_iter", [10]),
        num_eval_rollouts=trial.suggest_categorical("algo_num_eval_rollouts", [5]),
        warmstart=trial.suggest_categorical("algo_warmstart", [True]),
        thold_succ_subrtn=trial.suggest_categorical("algo_thold_succ_subrtn", [50]),
        subrtn_snapshot_mode="latest",
    )
    algo = SimOpt(study_dir, env_sim, env_real, subrtn_policy, subrtn_distr, **algo_hparam, logger=csv_logger)

    # Jeeeha
    algo.train(seed=args.seed)

    # Evaluate
    min_rollouts = 1000
    sampler = ParallelRolloutSampler(
        env_real, algo.policy, num_workers=1, min_rollouts=min_rollouts
    )  # parallelize via optuna n_jobs
    ros = sampler.sample()
    mean_ret = sum([r.undiscounted_return() for r in ros]) / min_rollouts

    return mean_ret


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    if args.dir is None:
        ex_dir = setup_experiment(
            "hyperparams", QQubeSwingUpSim.name, f"{SimOpt.name}-{CEM.name}_{QQubeSwingUpAndBalanceCtrl.name}_100Hz"
        )
        study_dir = osp.join(pyrado.TEMP_DIR, ex_dir)
        print_cbt(f"Starting a new Optuna study.", "c", bright=True)
    else:
        study_dir = args.dir
        if not osp.isdir(study_dir):
            raise pyrado.PathErr(given=study_dir)
        print_cbt(f"Continuing an existing Optuna study.", "c", bright=True)

    name = f"{QQubeSwingUpSim.name}_{SimOpt.name}-{CEM.name}_{QQubeSwingUpAndBalanceCtrl.name}_100Hz"
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
