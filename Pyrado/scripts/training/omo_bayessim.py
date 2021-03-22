"""
Simple script which runs SNPE-A with one fixed observation.
"""

import numpy as np
import torch as to
from copy import deepcopy
from sbi import utils

import pyrado
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.policies.special.mdn import MDNPolicy
from pyrado.utils.argparser import get_argparser

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(OneMassOscillatorSim.name, f"{BayesSim.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Define a mapping: index - domain parameter
    dp_mapping = {0: "m", 1: "k", 2: "d"}

    # Environments
    env_hparams = dict(dt=1 / 50.0, max_steps=200)
    env_sim = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))

    # Create a fake ground truth target domain
    env_real = deepcopy(env_sim)
    env_real.domain_param = dict(m=0.8, k=33, d=0.3)

    # Behavioral policy
    policy = IdlePolicy(env_sim.spec)

    # Prior
    dp_nom = env_sim.get_nominal_domain_param()  # m=1.0, k=30.0, d=0.5
    prior_hparam = dict(
        low=to.tensor([dp_nom["m"] * 0.5, dp_nom["k"] * 0.5, dp_nom["d"] * 0.5]),
        high=to.tensor([dp_nom["m"] * 1.5, dp_nom["k"] * 1.5, dp_nom["d"] * 1.5]),
    )
    prior = utils.BoxUniform(**prior_hparam)

    # Time series embedding
    embedding_hparam = dict(downsampling_factor=1)
    embedding = BayesSimEmbedding(env_sim.spec, RolloutSamplerForSBI.get_dim_data(env_sim.spec), **embedding_hparam)

    # Posterior (mixture of Gaussians) created inside BayesSim
    posterior_hparam = dict(num_comp=5, hidden_sizes=[42, 42], hidden_nonlin=to.relu, use_cuda=False)

    # Generate real_world observations
    num_real_rollouts = 1
    num_segments = 1
    # TODO delete below
    # rollout_worker = SimRolloutSamplerForSBI(env_sim, policy, dp_mapping, embedding, num_segments=num_segments)
    # dp_nom_to = to.tensor(list(dp_nom.values()))
    # data_real = to.stack([rollout_worker(dp_nom_to).squeeze() for _ in range(num_real_rollouts)])

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=300,
        num_segments=num_segments,
        posterior_hparam=posterior_hparam,
        num_sbi_rounds=1,
        subrtn_sbi_training_hparam=dict(
            max_iter=100,
            num_eval_samples=20,
            batch_size=50,
            max_grad_norm=5.0,
            lr=1e-3,
            use_gaussian_proposal=False,
        ),
        # num_workers=1,
    )

    algo = BayesSim(
        ex_dir,
        env_sim,
        env_real,
        policy,
        dp_mapping,
        prior,
        embedding,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy_name=policy.name),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior=posterior_hparam, posterior_name=MDNPolicy.name),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
