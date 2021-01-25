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
Train an agent to solve the Ball-on-Beam environment using Relative Entropy Search.

.. note::
    The hyper-parameters are not tuned at all!
"""
import pyrado
from pyrado.algorithms.episodic.reps import REPS
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnBeamSim.name, f"{REPS.name}_{LinearPolicy.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 100.0, max_steps=500)
    env = BallOnBeamSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        # feats=FeatureStack([RandFourierFeat(env.obs_space.flat_dim, num_feat=100, bandwidth=env.obs_space.bound_up)])
        # feats=FeatureStack([RBFFeat(num_feat_per_dim=20, bounds=env.obs_space.bounds, scale=0.8)]),
        feats=FeatureStack([identity_feat, sin_feat])
    )
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=500,
        eps=0.2,
        pop_size=10 * policy.num_param,
        num_init_states_per_domain=10,
        expl_std_init=0.2,
        expl_std_min=0.02,
        num_epoch_dual=1000,
        optim_mode="scipy",
        lr_dual=1e-3,
        use_map=True,
        num_workers=8,
    )
    algo = REPS(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
