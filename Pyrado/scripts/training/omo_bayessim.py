"""
Script to identify the domain parameters of the Pendulum environment using BayesSim
"""

from copy import deepcopy

import numpy as np
import torch as to
from sbi import utils

import pyrado
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_forward.dummy import IdlePolicy
from pyrado.sampling.sbi_embeddings import BayesSimEmbedding
from pyrado.utils.argparser import get_argparser
from pyrado.utils.sbi import create_embedding


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()
    seed_str, num_segs_str, len_seg_str = "", "", ""
    if args.seed is not None:
        seed_str = f"_seed-{args.seed}"
    if args.num_segments is not None:
        num_segs_str = f"numsegs-{args.num_segments}"
    elif args.len_segments is not None:
        len_seg_str = f"lensegs-{args.len_segments}"
    else:
        raise pyrado.ValueErr(msg="Either num_segments or len_segments must not be None, but not both or none!")

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(
        OneMassOscillatorSim.name,
        f"{BayesSim.name}_{IdlePolicy.name}",
        num_segs_str + len_seg_str + seed_str,
    )

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 100.0, max_steps=400)
    env_sim = OneMassOscillatorSim(**env_hparams, task_args=dict(task_args=dict(state_des=np.array([0.5, 0]))))

    # Create a fake ground truth target domain
    num_real_rollouts = 2
    env_real = deepcopy(env_sim)
    # randomizer = DomainRandomizer(
    #     NormalDomainParam(name="m", mean=0.8, std=0.8 / 50),
    #     NormalDomainParam(name="k", mean=33.0, std=33 / 50),
    #     NormalDomainParam(name="d", mean=0.3, std=0.3 / 50),
    # )
    # env_real = DomainRandWrapperBuffer(env_real, randomizer)
    # env_real.fill_buffer(num_real_rollouts)
    env_real.domain_param = dict(m=0.8, k=36, d=0.3)

    # Behavioral policy
    policy = IdlePolicy(env_sim.spec)

    # Define a mapping: index - domain parameter
    dp_mapping = {0: "m", 1: "k", 2: "d"}

    # Prior
    dp_nom = env_sim.get_nominal_domain_param()  # m=1.0, k=30.0, d=0.5
    prior_hparam = dict(
        low=to.tensor([dp_nom["m"] * 0.5, dp_nom["k"] * 0.5, dp_nom["d"] * 0.5]),
        high=to.tensor([dp_nom["m"] * 1.5, dp_nom["k"] * 1.5, dp_nom["d"] * 1.5]),
    )
    prior = utils.BoxUniform(**prior_hparam)

    # Time series embedding
    embedding_hparam = dict(downsampling_factor=1)
    embedding = create_embedding(BayesSimEmbedding.name, env_sim.spec, **embedding_hparam)

    # Posterior (mixture of Gaussians)
    posterior_hparam = dict(model="mdn", num_components=5)

    # Algorithm
    algo_hparam = dict(
        num_real_rollouts=num_real_rollouts,
        num_sim_per_round=500,
        num_segments=args.num_segments,
        len_segments=args.len_segments,
        num_sbi_rounds=4,
        num_eval_samples=100,
        subrtn_sbi_training_hparam=dict(
            training_batch_size=50,  # default: 50
            learning_rate=5e-4,  # default: 5e-4
            validation_fraction=0.2,  # default: 0.1
            stop_after_epochs=20,  # default: 20
            retrain_from_scratch_each_round=False,  # default: False
            show_train_summary=False,  # default: False
            # max_num_epochs=5,  # only use for debugging
        ),
        subrtn_policy=None,
        num_workers=20,
    )
    algo = BayesSim(
        save_dir=ex_dir,
        env_sim=env_sim,
        env_real=env_real,
        policy=policy,
        dp_mapping=dp_mapping,
        prior=prior,
        embedding=embedding,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(policy_name=policy.name),
        dict(prior=prior_hparam),
        dict(embedding=embedding_hparam, embedding_name=embedding.name),
        dict(posterior_nn=posterior_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
