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
Domain parameter identification experiment for a 2-dim Gaussian posterior as described in [1].
This experiment features ground-truth posterior samples which are generated using MCMC-sampling. Those are used to compare the samples from the approximate posterior.
The pre-computed ground-truth date is stored in 'data/perma/evaluation/2dg'.
Note that this script is supposed to work only on a single condition/real-rollout.

.. seealso::
    [1] G. Papamakarios, D. Sterratt, I. Murray, "Sequential Neural Likelihood: Fast Likelihood-free Inference with
        Autoregressive Flows", AISTATS, 2019
"""

import os.path as osp
from copy import deepcopy

import sbi.utils as utils
import torch as to
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environments.one_step.two_dim_gaussian import TwoDimGaussian
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.sbi_embeddings import LastStepEmbedding
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.utils.argparser import get_argparser


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    parser.add_argument(
        "--condition",
        dest="condition",
        type=int,
        default=1,
        help="Choose which real rollout should be used as a condition. You can choose between 1 and 10",
    )
    args = parser.parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(TwoDimGaussian.name, NPDR.name, f"observation_{args.condition}")

    # load and save the
    reference_posterior_samples = pyrado.load(
        f"reference_posterior_samples_{args.condition}.pt",
        osp.join(pyrado.EVAL_DIR, "2dg", f"observation_{args.condition}"),
    )
    true_param = pyrado.load(
        f"true_parameters_{args.condition}.pt", osp.join(pyrado.EVAL_DIR, "2dg", f"observation_{args.condition}")
    )
    pyrado.save(reference_posterior_samples, f"reference_posterior_samples.pt", ex_dir)
    pyrado.save(true_param, f"true_parameters.pt", ex_dir)

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim = TwoDimGaussian()
    env_real = osp.join(pyrado.EVAL_DIR, "2dg", f"observation_{args.condition}")

    # Behavioral policy
    policy = IdlePolicy(env_sim.spec)

    # Prior
    prior_hparam = dict(low=-3 * to.ones((5,)), high=3 * to.ones((5,)))
    prior = utils.BoxUniform(**prior_hparam)

    # Embedding
    embedding_hparam = dict()
    embedding = LastStepEmbedding(env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam)

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", hidden_features=50, num_transforms=5)

    # Algorithm
    dp_mapping = {0: "m_1", 1: "m_2", 2: "s_1", 3: "s_2", 4: "rho"}
    algo_hparam = dict(
        max_iter=1,
        num_sbi_rounds=5,
        num_real_rollouts=1,  # Should remain unchanged
        num_sim_per_round=1000,
        simulation_batch_size=10,
        num_segments=1,
        normalize_posterior=False,
        num_eval_samples=20,
        posterior_hparam=posterior_hparam,
        use_rec_act=True,
        subrtn_sbi_training_hparam=dict(
            num_atoms=10,  # default: 10
            training_batch_size=50,  # default: 50
            learning_rate=5e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            discard_prior_samples=False,  # default: False
            use_combined_loss=False,  # default: False
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
        ),
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=True),
        num_workers=1,
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
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(seed=args.seed),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        dict(condition=args.condition),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
