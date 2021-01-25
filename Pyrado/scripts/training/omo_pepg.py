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
Train an agent to solve the One-Mass-Oscillator environment using Parameter-Exploring Policy Gradients.
"""
import numpy as np

import pyrado
from pyrado.algorithms.episodic.pepg import PEPG
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.features import FeatureStack, const_feat, identity_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, f"{PEPG.name}_{LinearPolicy.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparams = dict(dt=1 / 50.0, max_steps=200)
    env = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))
    env = ActNormWrapper(env)

    # Policy
    policy_hparam = dict(feats=FeatureStack([const_feat, identity_feat]))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=100,
        num_init_states_per_domain=8,
        pop_size=60,
        expl_std_init=1.0,
        clip_ratio_std=0.05,
        normalize_update=False,
        transform_returns=True,
        lr=1e-2,
        num_workers=8,
    )
    algo = PEPG(ex_dir, env, policy, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy=policy_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
