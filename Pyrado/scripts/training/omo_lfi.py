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
from copy import deepcopy

import numpy as np
import torch as to
import torch.nn as nn
from sbi.inference import SNPE
from sbi import utils
from torch.distributions import MultivariateNormal

import pyrado
from pyrado.algorithms.inference.lfi2 import LFI
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.feed_forward.two_headed_fnn import TwoHeadedFNNPolicy
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, f"{LFI.name}_nflow")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 50.0, max_steps=200)
    env_sim = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))

    # Create a fake 'ground truth' target domain
    num_real_obs = 5
    env_real = deepcopy(env_sim)
    randomizer = DomainRandomizer(
        NormalDomainParam(name="k", mean=30.0, std=0.3),
        NormalDomainParam(name="d", mean=0.1, std=0.001),
    )
    env_real = DomainRandWrapperBuffer(env_real, randomizer)
    env_real.fill_buffer(num_real_obs)

    params_names = ["k", "d"]  # TODO replace

    # Policy
    behavior_policy = IdlePolicy(env_sim.spec)

    # Define a prior
    prior = utils.BoxUniform(low=to.tensor([27.0, 0.05]), high=to.tensor([33, 0.15]))

    # Normalizing flow
    embedding_net = nn.Identity()
    flow_hparam = dict(hidden_features=10, num_transforms=2)
    flow = utils.posterior_nn(model="maf", embedding_net=embedding_net, **flow_hparam)

    # Algorithm
    algo_hparam = dict(max_iter=5, num_sim=10, num_real_rollouts=num_real_obs)
    algo = LFI(
        ex_dir,
        env_sim,
        env_real,
        behavior_policy,
        params_names,
        prior,
        flow,
        SNPE,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_list_of_dicts_to_yaml(
        [
            dict(env=env_hparams, seed=args.seed),
            dict(flow=flow_hparam),
            dict(algo=algo_hparam, algo_name=algo.name),
        ],
        ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)

    # sample_params, _, _ = algo.evaluate(
    #     rollouts_real=ro_real, num_samples=num_samples, compute_quantity={"sample_params": True}
    # )
