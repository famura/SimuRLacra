import math

import pyrado
import torch as to
import numpy as np
from pyrado.environments.base import Env
from sbi.mcmc import SliceSampler

from tqdm import tqdm


def posterior_sampler(
    env,
    prior,
    observations,
    strategy: str = "rejection",
    params_names=None,
    start_param=None,
    num_samples=100,
    warmup=100,
):
    """
    Calculates posterior samples using MCMC sampling from a prior and a likelihood (Env).
    This function only works if the log-probability of a rollout can be calculated,
        see /pyrado/environments/one_step/multivariate_gaussian.py
    """
    if isinstance(env, Env) and params_names is None:
        raise pyrado.ValueErr(given=params_names)
    # if not initial start parameter, sample one from the prior
    prior_sample = prior.sample() if start_param is None else start_param

    # get posterior samples for each observation
    true_samples = []
    for obs in observations:
        # function which returns the log-probability of the unnormalized posterior
        if isinstance(env, Env):

            def lp_f(params):
                if isinstance(params, np.ndarray):
                    params = to.tensor(params)
                return (env.log_prob(obs, dict(zip(params_names, params))) + prior.log_prob(params)).item()

        else:

            def lp_f(params):
                if isinstance(params, np.ndarray):
                    params = to.tensor(params, dtype=to.float32)
                return (env(obs).log_prob(params) + prior.log_prob(params)).item()

        sampler = SliceSampler(lp_f=lp_f, x=prior_sample)
        _ = sampler.gen(warmup)

        true_samples.append(to.as_tensor(sampler.gen(num_samples), dtype=to.float32))
    return to.stack(true_samples)


def rejection_sampler(
    prior,
    likelihood,
    observations,
    num_samples,
    proposal_dist=None,
    true_params=None,
    num_batches_without_new_max: int = 1000,
    batch_size: int = 100,
    multiplier_m: float = 1.1,
):
    # TODO: WIP, does work well on simple Distributions, but lacks when it gets more difficult
    # TODO: check if multiple observations are given

    if proposal_dist is None:
        proposal_dist = prior

    parameters = true_params
    samples_for_each_obs = []
    for obs in observations:
        log_m = -float("inf")
        if isinstance(likelihood, Env):
            log_likelihood = lambda param: likelihood.log_prob(obs, param)
        else:
            log_likelihood = lambda param: likelihood(obs).log_prob(param)

        pbar = tqdm(range(num_batches_without_new_max))
        num_batches_cnt = 0
        while num_batches_cnt <= num_batches_without_new_max:
            if parameters is None:
                parameters = prior.sample((batch_size,))
            log_prob_likelihood_batch = log_likelihood(parameters)

            # log_prob_proposal_batch = proposal_dist.log_prob(parameters)
            # log_prob_ratio_max = (log_prob_likelihood_batch - log_prob_proposal_batch).max()
            log_prob_ratio_max = log_prob_likelihood_batch.max()

            if log_prob_ratio_max > log_m:
                log_m = log_prob_ratio_max + math.log(multiplier_m)
                num_batches_cnt = 0
                pbar.reset()
                pbar.set_postfix_str(s=f"log(M): {log_m:.3f}", refresh=True)
            else:
                num_batches_cnt += 1
                pbar.update()

            parameters = None

        num_sims = 0
        num_accepted = 0
        samples = []
        pbar = tqdm(total=num_samples)
        while num_accepted < num_samples:
            u = to.rand((batch_size,))
            proposal = proposal_dist.sample((batch_size,))
            # probs = log_likelihood(proposal) - (log_m + proposal_dist.log_prob(proposal))
            probs = log_likelihood(proposal).squeeze() - log_m
            num_sims += batch_size
            accept_idxs = to.where(probs > to.log(u))[0]
            num_accepted += len(accept_idxs)

            if len(accept_idxs) > 0:
                samples.append(proposal[accept_idxs].detach())
                pbar.update(len(accept_idxs))
                pbar.set_postfix_str(s=f"Acceptance rate: {num_accepted / num_sims:.9f}", refresh=True)

        pbar.close()

        samples = to.cat(samples)[:num_samples, :]
        assert samples.shape[0] == num_samples
        samples_for_each_obs.append(samples)
    samples_for_each_obs = to.stack(samples_for_each_obs)

    return samples_for_each_obs
