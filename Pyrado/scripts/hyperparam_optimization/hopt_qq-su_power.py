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
Optimize the hyper-parameters of Policy learning by Weighting Exploration with the Returns
for the Quanser Qube swing-up task.
"""
import functools
import os
import os.path as osp

import optuna
from optuna.pruners import MedianPruner

import pyrado
from pyrado.algorithms.episodic.power import PoWER
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.logger.step import create_csv_step_logger
from pyrado.policies.features import (
    ATan2Feat,
    FeatureStack,
    MultFeat,
    abs_feat,
    cubic_feat,
    identity_feat,
    sign_feat,
    squared_feat,
)
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.sampling.parallel_rollout_sampler import ParallelRolloutSampler
from pyrado.utils.argparser import get_argparser
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
    env_hparams = dict(dt=1 / 250.0, max_steps=1500)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack(
            [identity_feat, sign_feat, abs_feat, squared_feat, cubic_feat, ATan2Feat(1, 2), MultFeat((4, 5))]
        )
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        num_workers=1,  # parallelize via optuna n_jobs
        max_iter=50,
        pop_size=trial.suggest_int("pop_size", 50, 200),
        num_init_states_per_domain=trial.suggest_int("num_init_states_per_domain", 4, 10),
        num_is_samples=trial.suggest_int("num_is_samples", 5, 40),
        expl_std_init=trial.suggest_uniform("expl_std_init", 0.1, 0.5),
        symm_sampling=trial.suggest_categorical("symm_sampling", [True, False]),
    )
    csv_logger = create_csv_step_logger(osp.join(study_dir, f"trial_{trial.number}"))
    algo = PoWER(osp.join(study_dir, f"trial_{trial.number}"), env, policy, **algo_hparam, logger=csv_logger)

    # Train without saving the results
    algo.train(snapshot_mode="latest", seed=seed)

    # Evaluate
    min_rollouts = 1000
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
        ex_dir = setup_experiment(
            "hyperparams", QQubeSwingUpSim.name, f"{PoWER.name}_{LinearPolicy.name}_250Hz_actnorm"
        )
        study_dir = osp.join(pyrado.TEMP_DIR, ex_dir)
        print_cbt(f"Starting a new Optuna study.", "c", bright=True)
    else:
        study_dir = args.dir
        if not osp.isdir(study_dir):
            raise pyrado.PathErr(given=study_dir)
        print_cbt(f"Continuing an existing Optuna study.", "c", bright=True)

    name = f"{QQubeSwingUpSim.name}_{PoWER.name}_{LinearPolicy.name}_250Hz_actnorm"
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
