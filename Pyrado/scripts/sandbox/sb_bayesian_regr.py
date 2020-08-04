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
Test Bayesian Regression using Pyro and the One Mass Oscillator setup.
See https://pyro.ai/examples/svi_part_i.html
"""
import pyro
import pyro.distributions as distr
import pyro.optim as optim
import sys
import torch as to
from matplotlib import pyplot as plt
from pyro.infer import SVI, Trace_ELBO
from torch.distributions import constraints
from tqdm import tqdm

from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim, OneMassOscillatorDyn
from pyrado.policies.dummy import DummyPolicy
from pyrado.sampling.parallel_sampler import ParallelSampler


def model(states, actions, observations, prior):
    """
    .. note::
        `pyro.plate` assumes that the observations are conditionally independent given the latents.
        `pyro.plate` is not appropriate for temporal models where each iteration of a loop depends on the previous
        iteration. In this case a `range` or `pyro.markov` should be used instead.

    :param states:
    :param actions:
    :param observations:
    :param prior:
    :return:
    """
    # Priors (also defines the support)
    m = pyro.sample('m', distr.Normal(prior['m'], prior['m']/6.))
    k = pyro.sample('k', distr.Normal(prior['k'], prior['k']/6.))
    d = pyro.sample('d', distr.Normal(prior['d'], prior['d']/4.))
    sigma = pyro.sample('sigma', distr.Normal(5., 0.5))  # obs noise scale; these params seem to have no effect

    # Create a model for learning the domain parameters
    omo_dyn = OneMassOscillatorDyn(dt=dt)

    with pyro.plate('data_loop'):  # no len(observations) needed for vectorized pyro.plate
        # Likelihood conditioned on m, k, d (sampled outside the loop)
        pyro.sample('obs', distr.Normal(omo_dyn(states, actions, dict(m=m, k=k, d=d)),
                                        sigma.clamp_(min=to.tensor(1e-3))).to_event(1),
                    obs=observations)


def guide(states, actions, observations, prior):
    """
    The guide serves as an approximation to the posterior.
    The guide does not contain observed data, since the guide needs to be a properly normalized distribution.
    The distributions used in the guide can be different from the ones used in the model, but the random variable names
    must be identical.

    .. note::
        The guide must not contain `pyro.sample` statements with the `obs` argument.

    :param states:
    :param actions:
    :param observations:
    :param prior:
    """
    m_loc = pyro.param('m_loc', to.tensor(prior['m']), constraint=constraints.positive)
    m_scale = pyro.param('m_scale', to.tensor(prior['m']/6.), constraint=constraints.positive)
    k_loc = pyro.param('k_loc', to.tensor(prior['k']), constraint=constraints.positive)
    k_scale = pyro.param('k_scale', to.tensor(prior['k']/6.), constraint=constraints.positive)
    d_loc = pyro.param('d_loc', to.tensor(prior['d']), constraint=constraints.positive)
    d_scale = pyro.param('d_scale', to.tensor(prior['d']/6.), constraint=constraints.positive)
    sigma_loc = pyro.param('sigma_loc', to.tensor(1.0), constraint=constraints.positive)

    pyro.sample('m', distr.Normal(m_loc, m_scale))
    pyro.sample('k', distr.Normal(k_loc, k_scale))
    pyro.sample('d', distr.Normal(d_loc, d_scale))
    pyro.sample('sigma', distr.Normal(sigma_loc, to.tensor(0.001)))  # scale is a sensitive parameter; < 1e-3


def train(svi, rollouts, prior, num_epoch=2000, print_iter=100):
    # A fresh new start
    pyro.clear_param_store()

    # Prepare the data
    [ro.torch(data_type=to.get_default_dtype()) for ro in rollouts]
    states_cat = to.cat([ro.observations[:-1] for ro in rollouts])
    actions_cat = to.cat([ro.actions for ro in rollouts])
    targets_cat = to.cat([(ro.observations[1:] - ro.observations[:-1]) for ro in rollouts])  # state deltas

    elbo_hist = []
    m_loc_hist, m_scale_hist = [], []
    k_loc_hist, k_scale_hist = [], []
    d_loc_hist, d_scale_hist = [], []
    sigma_hist = []

    # Train
    for i in tqdm(range(num_epoch), total=num_epoch, desc='Training', unit='epochs', file=sys.stdout, leave=False):
        # The args of step are forwarded to the model and the guide
        elbo_hist.append(svi.step(states_cat, actions_cat, targets_cat, prior))

        m_loc_hist.append(pyro.param('m_loc').item())
        m_scale_hist.append(pyro.param('m_scale').item())
        k_loc_hist.append(pyro.param('k_loc').item())
        k_scale_hist.append(pyro.param('k_scale').item())
        d_loc_hist.append(pyro.param('d_loc').item())
        d_scale_hist.append(pyro.param('d_scale').item())
        sigma_hist.append(pyro.param('sigma_loc').item())

    for name, value in pyro.get_param_store().items():
        print(f'param: {name} \t value: {pyro.param(name)}')

    # Plot
    _, axs = plt.subplots(4, 2, figsize=(12, 8))
    axs[0, 0].plot(elbo_hist)
    axs[0, 0].set_xlabel('step')
    axs[0, 0].set_ylabel('ELBO loss')

    axs[0, 1].plot(m_loc_hist)
    axs[0, 1].plot([0, num_epoch], [dp_gt['m'], dp_gt['m']], c='k')
    axs[0, 1].set_ylabel('m\_loc')

    axs[1, 0].plot(k_loc_hist)
    axs[1, 0].plot([0, num_epoch], [dp_gt['k'], dp_gt['k']], c='k')
    axs[1, 0].set_ylabel('k\_loc')

    axs[1, 1].plot(d_loc_hist)
    axs[1, 1].plot([0, num_epoch], [dp_gt['d'], dp_gt['d']], c='k')
    axs[1, 1].set_ylabel('d\_loc')

    axs[2, 0].plot(m_scale_hist)
    axs[2, 0].set_ylabel('m\_scale')

    axs[2, 1].plot(k_scale_hist)
    axs[2, 1].set_ylabel('k\_scale')

    axs[3, 0].plot(d_scale_hist)
    axs[3, 0].set_ylabel('d\_scale')

    axs[3, 1].plot(sigma_hist)
    axs[3, 1].set_ylabel('obs\_noise\_scale')

    plt.show()


if __name__ == '__main__':
    # Set up environment
    dp_gt = dict(m=2., k=20., d=0.8)  # ground truth
    dp_init = dict(m=1.0, k=24., d=0.4)  # initial guess
    dt = 1/50.
    env = OneMassOscillatorSim(dt=dt, max_steps=400)
    env.reset(domain_param=dp_gt)

    # Set up policy
    policy = DummyPolicy(env.spec)

    # Sample
    sampler = ParallelSampler(env, policy, num_workers=1, min_rollouts=50, seed=1)
    ros = sampler.sample()

    # Pyro
    pyro.set_rng_seed(1001)
    pyro.enable_validation(True)

    train(
        SVI(model=model,
            guide=guide,
            optim=optim.Adam({'lr': 0.01}),
            # optim=optim.SGD({'lr': 0.001, 'momentum': 0.1}),
            loss=Trace_ELBO()),
        rollouts=ros, prior=dp_init
    )
