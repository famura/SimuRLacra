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
Train an agent to solve the Ball-on-Plate environment using Soft Actor-Critic.

.. note::
    The hyper-parameters are not tuned at all!
"""
import torch as to

import pyrado
from pyrado.algorithms.step_based.sac import SAC
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.rcspysim.ball_on_plate import BallOnPlate2DSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.policies.feed_back.two_headed_fnn import TwoHeadedFNNPolicy
from pyrado.spaces import BoxSpace, ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(BallOnPlate2DSim.name, f"{SAC.name}_{TwoHeadedFNNPolicy.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(physicsEngine="Bullet", dt=1 / 100.0, max_steps=500)
    env = BallOnPlate2DSim(**env_hparams)
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(
        shared_hidden_sizes=[32, 32],
        shared_hidden_nonlin=to.relu,
    )
    policy = TwoHeadedFNNPolicy(spec=env.spec, **policy_hparam)

    # Critic
    qfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.relu)
    obsact_space = BoxSpace.cat([env.obs_space, env.act_space])
    qfcn_1 = FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **qfcn_hparam)
    qfcn_2 = FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **qfcn_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1000 * env.max_steps,
        memory_size=1000 * env.max_steps,
        gamma=0.995,
        num_updates_per_step=1,
        tau=0.99,
        ent_coeff_init=0.2,
        learn_ent_coeff=False,
        target_update_intvl=1,
        standardize_rew=False,
        min_steps=1,
        batch_size=256,
        num_workers=4,
        lr=3e-4,
    )
    algo = SAC(ex_dir, env, policy, qfcn_1, qfcn_2, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(qfcn=qfcn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
