import torch as to
import numpy as np
from sbi.mcmc import SliceSampler


def posterior_sampler(env,
                      prior,
                      observations,
                      params_names,
                      start_param=None,
                      num_samples=100,
                      warmup=100,
                      ):
    """
    Calculates posterior samples using MCMC sampling from a prior and a likelihood (Env).
    This function only works if the log-probability of a rollout can be calculated,
        see /pyrado/environments/one_step/multivariate_gaussian.py
    """
    # if not initial start parameter, sample one from the prior
    prior_sample = prior.sample() if start_param is None else start_param

    # get posterior samples for each observation
    true_samples = []
    for obs in observations:

        # function which returns the log-probability of the unnormalized posterior
        def lp_f(params):
            if isinstance(params, np.ndarray):
                params = to.tensor(params)

            return (env.log_prob(obs, dict(zip(params_names, params))) + prior.log_prob(params)).item()

        sampler = SliceSampler(lp_f=lp_f, x=prior_sample)
        _ = sampler.gen(warmup)

        true_samples.append(to.as_tensor(sampler.gen(num_samples), dtype=to.float32))
    return to.stack(true_samples)
