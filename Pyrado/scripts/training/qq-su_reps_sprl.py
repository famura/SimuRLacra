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
Train an agent to solve the Quanser Qube swing-up task using Relative Entropy Search.
"""
from multiprocessing import cpu_count

import pyrado
import torch as to
from pyrado.algorithms.episodic.reps import REPS
from pyrado.algorithms.meta.sprl import SPRL
from pyrado.domain_randomization.domain_parameter import SelfPacedLearnerParameter
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.features import (
    FeatureStack,
    identity_feat,
    sign_feat,
    abs_feat,
    squared_feat,
    const_feat,
    MultFeat,
    ATan2Feat,
    cubic_feat,
)
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.utils.argparser import get_argparser

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{REPS.name}_{LinearPolicy.name}_sprl")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 250.0, max_steps=12 * 250)
    env = QQubeSwingUpSim(**env_hparams)
    env = ActNormWrapper(env)
    env = DomainRandWrapperLive(
        env,
        randomizer=DomainRandomizer(
            SelfPacedLearnerParameter(
                name="g",
                target_mean=to.tensor([9.81]),
                target_cov_chol_flat=to.tensor([1.0]),
                context_mean=to.tensor([9.81]),
                context_cov_chol_flat=to.tensor([0.01]),
            )
        ),
    )

    # Policy
    policy_hparam = dict(
        # feats=FeatureStack([RandFourierFeat(env.obs_space.flat_dim, num_feat_per_dim=20, bandwidth=env.obs_space.bound_up)])
        feats=FeatureStack(
            [
                const_feat,
                identity_feat,
                sign_feat,
                abs_feat,
                squared_feat,
                cubic_feat,
                ATan2Feat(1, 2),
                MultFeat((3, 4)),
            ]
        )
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=20,
        eps=1.0,
        pop_size=20 * policy.num_param,
        num_init_states_per_domain=4,
        expl_std_init=0.2,
        expl_std_min=0.02,
        use_map=True,
        optim_mode="scipy",
        num_workers=cpu_count(),
    )
    algo = REPS(ex_dir, env, policy, **algo_hparam)
    sprl_hparam = dict(
        kl_constraints_ub=8000,
        performance_lower_bound=500,
        std_lower_bound=0.4,
        kl_threshold=200,
        max_iter=50,
    )
    algo = SPRL(env, algo, **sprl_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="best", seed=args.seed)
