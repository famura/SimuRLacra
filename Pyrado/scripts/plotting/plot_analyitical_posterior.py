"""
Script to evaluate a posterior obtained using the sbi package.
Comparison is carried out via Maximum-Mean Discrepancy.
This script will only be successful if the simulator is tractable i.e. its probability can be evaluated.
"""
import os
import os.path as osp
import torch as to
import pandas as pd
from matplotlib import pyplot as plt

import pyrado
from matplotlib.colors import DivergingNorm
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.inference.lfi import LFI
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapper
from pyrado.logger.experiment import ask_for_experiment
from pyrado.plotting.distribution import draw_posterior_distr
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.utils.argparser import get_argparser
from pyrado.utils.experiments import load_experiment

from pyrado.sampling.posterior_sampler import posterior_sampler, rejection_sampler

from geomloss.samples_loss import SamplesLoss

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Get the experiment's directory to load from
    ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # Load the environments, the policy, and the posterior
    env_sim, policy, kwout = load_experiment(ex_dir, args)

    env_real = pyrado.load(None, "env_real", "pkl", ex_dir)
    # check if environment is wrapped into DomainRandWrapperBuffer
    env_real = env_real.wrapped_env if isinstance(env_real, DomainRandWrapper) else env_real

    prior = kwout["prior"]
    posterior = kwout["posterior"]

    # Load the algorithm and the required data
    algo = Algorithm.load_snapshot(ex_dir)
    if not isinstance(algo, LFI):
        raise pyrado.TypeErr(given=algo, expected_type=LFI)

    # algo hparams
    params_names = list(algo.dp_mapping.values())
    num_sim_per_real_rollout = algo.num_sim_per_real_rollout
    num_real_rollouts = algo.num_real_rollouts

    reference_dir = osp.join(ex_dir, "../../../../perma/environment/ToyExample/reference")
    if not osp.isdir(reference_dir):
        raise pyrado.PathErr(given=reference_dir)

    # get samples from the true postserior
    # samples are from sbi benchmark and were generated using rejection sampling
    num_observation = 10
    reference_samples = [
        pyrado.load(None, "reference_posterior_samples", "pt", osp.join(reference_dir, f"num_observation_{n}"))
        for n in range(1, num_observation + 1)
    ]
    reference_samples = to.stack(reference_samples)[:, : args.num_samples, :]
    observation = [
        pyrado.load(None, "observation", "pt", osp.join(reference_dir, f"num_observation_{n}"))
        for n in range(1, num_observation + 1)
    ]
    observation = to.stack(observation)
    true_params = [
        pyrado.load(None, "true_parameters", "pt", osp.join(reference_dir, f"num_observation_{n}"))
        for n in range(1, num_observation + 1)
    ]
    true_params = to.stack(true_params)

    # read evaluated data
    check_files = ["num_samples", "mmd_mean", "mmd_std", "posterior_samples"]
    num_samples, mmd_mean, mmd_std, posterior_samples = [pyrado.load(None, cf, "pt", ex_dir) for cf in check_files]

    num_samples = to.tensor(num_samples, dtype=to.float32)
    mmd_mean = to.tensor(mmd_mean, dtype=to.float32)
    mmd_std = to.tensor(mmd_std, dtype=to.float32)
    posterior_samples = posterior_samples[-1]

    # plot trainings progress with mmd loss
    plt.figure()
    plt.plot(num_samples, mmd_mean)
    plt.fill_between(num_samples, mmd_mean - mmd_std, mmd_mean + mmd_std, color="gray", alpha=0.2)
    plt.xlabel("Number of samples")
    plt.ylabel("MMD-Loss")
    plt.savefig(osp.join(ex_dir, "mmd_loss.pdf"))
    plt.title("Loss")
    plt.show()

    which_obs = 0
    # plot the first two domain parameter
    dp_idcs = (0, 1)
    # set condition
    condition = posterior_samples[0].mean(dim=0)
    # plot the contourplot for the approximate posterior
    fig, axs = plt.subplots(1, 1, figsize=(14, 7), tight_layout=True)
    # _ = draw_posterior_distr(
    #     axs,
    #     "joint",
    #     posterior,
    #     observation[which_obs].unsqueeze(0),
    #     algo.dp_mapping,
    #     env_real,
    #     prior,
    #     dp_idcs,
    #     condition,
    #     show_prior=False,
    #     contourf_kwargs=dict(cmap="RdGy"),
    # )
    # plot reference posterior samples
    plt.scatter(reference_samples[which_obs, :, 0], reference_samples[which_obs, :, 1], color="black")
    plt.scatter(posterior_samples[which_obs, :, 0], posterior_samples[which_obs, :, 1], color="blue")
    plt.title("True vs. approximate Posterior")
    plt.savefig(osp.join(ex_dir, f"scatter_true_posterior_{which_obs}.pdf"))
    plt.show()
