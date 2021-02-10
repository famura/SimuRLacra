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
Sim-to-sim experiment on the Pendulum environment using likelihood-free inference
"""
import os.path as osp
import torch as to
import torch.nn as nn
from copy import deepcopy
from sbi.inference import SNPE
from sbi import utils
from torch.optim import lr_scheduler

import pyrado
from pyrado.algorithms.inference.lfi import LFI
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO2
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.feed_forward.fnn import FNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, f"{LFI.name}-{PPO2.name}_{FNNPolicy.name}")
    num_workers = 4

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=1000)
    env_sim = PendulumSim(**env_hparams)
    # env_sim.domain_param = dict(d_pole=0, tau_max=10.0)
    env_real = deepcopy(env_sim)

    # Create a fake ground truth target domain
    num_real_obs = 3
    env_real.domain_param = dict(m_pole=0.25, l_pole=2.0)
    dp_mapping = {0: "m_pole", 1: "l_pole"}

    # Prior and Posterior (normalizing flow)
    prior_hparam = dict(low=to.tensor([0.125, 1.0]), high=to.tensor([0.5, 4.0]))
    prior = utils.BoxUniform(**prior_hparam)
    posterior_nn_hparam = dict(model="maf", embedding_net=nn.Identity(), hidden_features=10, num_transforms=5)

    # Policy
    policy_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.relu)
    policy = FNNPolicy(spec=env_sim.spec, **policy_hparam)
    pyrado.load(policy, "policy", "pt", osp.join(pyrado.EXP_DIR, "pend", "ppo2_fnn", "2021-01-21_13-49-49--actnorm"))

    # Critic
    vfcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    vfcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **vfcn_hparam)
    pyrado.load(vfcn, "vfcn", "pt", osp.join(pyrado.EXP_DIR, "pend", "ppo2_fnn", "2021-01-21_13-49-49--actnorm"))
    critic_hparam = dict(
        gamma=0.9852477569514027,
        lamda=0.9729014682749334,
        num_epoch=5,
        batch_size=500,
        lr=2.7189235593899743e-3,
        max_grad_norm=5.0,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
    )
    critic = GAE(vfcn, **critic_hparam)

    # Policy optimization subroutine
    subrtn_policy_hparam = dict(
        max_iter=250,
        min_steps=30 * env_sim.max_steps,
        num_epoch=5,
        vfcn_coeff=1.190454086194093,
        entropy_coeff=4.944111681414721e-05,
        eps_clip=0.09657039413812532,
        batch_size=500,
        std_init=0.1,
        lr=8.775532791215318e-4,
        max_grad_norm=None,
        lr_scheduler=lr_scheduler.ExponentialLR,
        lr_scheduler_hparam=dict(gamma=0.999),
        num_workers=num_workers,
    )
    subrtn_policy = PPO2(ex_dir, env_sim, policy, critic, **subrtn_policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=10,
        summary_statistic="dtw_distance",  # bayessim or dtw_distance
        sbi_training_hparam=dict(learning_rate=3e-4),
        num_real_rollouts=num_real_obs,
        num_sim_per_real_rollout=500,
        normalize_posterior=False,
        num_eval_samples=100,
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
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(posterior_nn=posterior_nn_hparam),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, vfcn=vfcn_hparam),
        dict(subrtn_policy=subrtn_policy_hparam, subrtn_policy_name=subrtn_policy.name),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
