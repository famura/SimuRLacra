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
Train an agent to solve the Pendulum environment using Neural Posterior Domain Randomization
"""
from copy import deepcopy

import torch as to
from sbi import utils
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.episodic.hc import HCNormal
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.environments.pysim.pendulum import PendulumSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.features import FeatureStack, MultFeat, const_feat, identity_feat, sign_feat, squared_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.sampling.sbi_embeddings import DeltaStepsEmbedding, DynamicTimeWarpingEmbedding
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(PendulumSim.name, f"{NPDR.name}-{HCNormal.name}_{LinearPolicy.name}")
    num_workers = 8

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 50.0, max_steps=400)
    env_sim = PendulumSim(**env_hparams)
    # env_sim.domain_param = dict(d_pole=0, tau_max=10.0)

    # Create a fake ground truth target domain
    env_real = deepcopy(env_sim)
    env_real.domain_param = dict(m_pole=1 / 1.2 ** 2, l_pole=1.2)

    # Define a mapping: index - domain parameter
    dp_mapping = {0: "m_pole", 1: "l_pole"}

    # Prior
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        low=to.tensor([dp_nom["m_pole"] * 0.3, dp_nom["l_pole"] * 0.3]),
        high=to.tensor([dp_nom["m_pole"] * 1.7, dp_nom["l_pole"] * 1.7]),
    )
    prior = utils.BoxUniform(**prior_hparam)

    # Time series embedding
    # embedding_hparam = dict()
    # embedding = LastStepEmbedding(env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam)
    embedding_hparam = dict(downsampling_factor=5)
    embedding = DeltaStepsEmbedding(
        env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), env_sim.max_steps, **embedding_hparam
    )
    # embedding_hparam = dict(downsampling_factor=1)
    # embedding = BayesSimEmbedding(env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam)
    # embedding_hparam = dict(downsampling_factor=2)
    # embedding = DynamicTimeWarpingEmbedding(
    #     env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam
    # )
    # embedding_hparam = dict(hidden_size=10, num_recurrent_layers=2, output_size=1, downsampling_factor=10)
    # embedding = RNNEmbedding(
    #     env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), env_sim.max_steps, **embedding_hparam
    # )

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=20, num_transforms=4)

    # Policy
    policy_hparam = dict(
        feats=FeatureStack([const_feat, identity_feat, sign_feat, squared_feat, MultFeat((0, 2)), MultFeat((1, 2))])
    )
    policy = LinearPolicy(spec=env_sim.spec, **policy_hparam)

    # Policy optimization subroutine
    subrtn_policy_hparam = dict(
        max_iter=5,
        pop_size=5 * policy.num_param,
        num_domains=20,
        num_init_states_per_domain=1,
        expl_factor=1.05,
        expl_std_init=1.0,
        num_workers=num_workers,
    )
    subrtn_policy = HCNormal(ex_dir, env_sim, policy, **subrtn_policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=5,
        num_real_rollouts=1,
        num_sim_per_round=300,
        num_sbi_rounds=3,
        simulation_batch_size=10,
        normalize_posterior=False,
        num_eval_samples=100,
        # num_segments=1,
        len_segments=100,
        posterior_hparam=posterior_hparam,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=5e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=True,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=False),
        num_workers=num_workers,
    )
    algo = NPDR(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        SNPE_C,
        embedding,
        **algo_hparam,
        subrtn_policy=subrtn_policy,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(posterior_nn=posterior_hparam),
        dict(policy=policy_hparam),
        dict(subrtn_policy=subrtn_policy_hparam, subrtn_policy_name=subrtn_policy.name),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
