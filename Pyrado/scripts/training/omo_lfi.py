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
from copy import deepcopy

import numpy as np
import torch as to
import torch.nn as nn
from sbi.inference import SNPE
from sbi import utils

import pyrado
from pyrado.algorithms.inference.lfi2 import LFI
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, f"{LFI.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 50.0, max_steps=200)
    env_sim = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))

    # Create a fake 'ground truth' target domain
    num_real_obs = 1
    env_real = deepcopy(env_sim)
    randomizer = DomainRandomizer(
        NormalDomainParam(name="k", mean=33.0, std=33 / 50),
        NormalDomainParam(name="d", mean=0.2, std=0.2 / 50),
    )
    env_real = DomainRandWrapperBuffer(env_real, randomizer)
    env_real.fill_buffer(num_real_obs)
    dp_mapping = {0: "k", 1: "d"}

    # Policy
    behavior_policy = IdlePolicy(env_sim.spec)

    # Prior
    prior_hparam = dict(low=to.tensor([25.0, 0.05]), high=to.tensor([35, 0.45]))
    prior = utils.BoxUniform(**prior_hparam)

    # Posterior (normalizing flow)
    posterior_nn_hparam = dict(model="maf", embedding_net=nn.Identity(), hidden_features=10, num_transforms=2)

    # Algorithm
    algo_hparam = dict(
        summary_statistic="ramos",
        max_iter=15,
        num_real_rollouts=num_real_obs,
        num_sim_per_real_rollout=100,
        num_workers=1,
    )
    algo = LFI(
        ex_dir,
        env_sim,
        env_real,
        behavior_policy,
        dp_mapping,
        prior,
        posterior_nn_hparam,
        SNPE,
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

    # sample_params, _, _ = algo.evaluate(
    #     rollouts_real=ro_real, num_samples=num_samples, compute_quantity={"sample_params": True}
    # )
