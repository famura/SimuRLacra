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
Sim-to-sim experiment on the One-Mass-Oscillator environment using likelihood-free inference
"""

import torch as to
import torch.nn as nn
from copy import deepcopy
from sbi.inference import SNPE
from sbi import utils

import pyrado
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.inference.lfi2 import LFI
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, squared_feat, const_feat, sin_feat, cos_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, f"{LFI.name}")
    num_workers = 1

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=1000)
    env_sim = PendulumSim(**env_hparams)
    env_sim.domain_param = dict(d_pole=0, tau_max=5.0)

    # Create a fake ground truth target domain
    num_real_obs = 1
    env_real = deepcopy(env_sim)
    env_real.domain_param = dict(m_pole=0.25, l_pole=2.0)
    dp_mapping = {0: "m_pole", 1: "l_pole"}

    # Policy
    feats = FeatureStack([const_feat, identity_feat, sin_feat, cos_feat, squared_feat])
    policy = LinearPolicy(env_sim.spec, feats)

    # Policy optimization subroutine
    subrtn_policy_hparam = dict(
        max_iter=5,
        pop_size=10,
        num_rollouts=4,
        num_is_samples=10,
        expl_std_init=1e0,
        expl_std_min=1e-2,
        extra_expl_std_init=1e0,
        extra_expl_decay_iter=5,
        num_workers=num_workers,
    )
    subrtn_policy = CEM(ex_dir, env_sim, policy, **subrtn_policy_hparam)

    # Prior and Posterior (normalizing flow)
    prior_hparam = dict(low=to.tensor([0.0625, 0.0625]), high=to.tensor([4.0, 4.0]))
    prior = utils.BoxUniform(**prior_hparam)
    posterior_nn_hparam = dict(model="maf", embedding_net=nn.Identity(), hidden_features=10, num_transforms=2)

    # Algorithm
    algo_hparam = dict(
        summary_statistic="ramos",
        max_iter=15,
        num_real_rollouts=num_real_obs,
        num_sim_per_real_rollout=50,
        num_workers=num_workers,
    )
    algo = LFI(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        posterior_nn_hparam,
        SNPE,
        subrtn_policy=subrtn_policy,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml(
        [
            dict(env=env_hparams, seed=args.seed),
            dict(prior=prior_hparam),
            dict(posterior_nn=posterior_nn_hparam),
            dict(algo=algo_hparam, algo_name=algo.name),
        ],
        ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
