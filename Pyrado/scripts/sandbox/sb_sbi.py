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
Testing the simulation-based inference (SBI) toolbox

.. seealso::
    https://astroautomata.com/blog/simulation-based-inference/
"""
import numpy as np
import sbi.utils as utils
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sbi.inference.base import infer

plt.rcParams.update({"text.usetex": True})


def simulator(mu):
    # Generate samples from N(mu, sigma=0.5)
    return mu + 0.5 * torch.randn_like(mu)


if __name__ == '__main__':
    prior = utils.BoxUniform(
        low=torch.tensor([-5., -5.]),
        high=torch.tensor([5., 5.])
    )

    # Let’s learn a likelihood from the simulator
    num_sim = 200
    method = 'SNRE'  # SNPE or SNLE or SNRE
    posterior = infer(
        simulator,
        prior,
        method=method,  # SNRE newer than SNLE newer than SNPE
        num_workers=-1,
        num_simulations=num_sim)

    # Let’s record our 5 “observations” of the true distribution
    n_observations = 5
    observation = torch.tensor([3., -1.5])[None] + 0.5*torch.randn(n_observations, 2)

    # Inference
    samples = posterior.sample((200,), x=observation[0])  # sample the posterior for a single data point

    # Computing the log-probability
    bounds = [3-1, 3+1, -1.5-1, -1.5+1]
    mu_1, mu_2 = torch.tensor(np.mgrid[bounds[0]:bounds[1]:2/50., bounds[2]:bounds[3]:2/50.]).float()
    grids = torch.cat(
        (mu_1.reshape(-1, 1), mu_2.reshape(-1, 1)),
        dim=1
    )
    if method == 'SNPE':
        log_prob = sum([
            posterior.log_prob(grids, observation[i])
            for i in range(len(observation))
        ])
    else:
        log_prob = sum([
            posterior.net(torch.cat((grids, observation[i].repeat((grids.shape[0])).reshape(-1, 2)), dim=1))[:, 0] +
            posterior._prior.log_prob(grids)
            for i in range(len(observation))
        ]).detach()
    prob = torch.exp(log_prob - log_prob.max())  # scale the probabilities to [0, 1]

    log_probability = posterior.log_prob(samples, x=observation[0])
    out = utils.pairplot(samples, limits=[[-5,5],[-5,5]], fig_size=(6,6), upper='kde', diag='kde')

    true_like = lambda x: -((x[0] - mu_1)**2 + (x[1] - mu_2)**2)/(2*0.5**2)
    true_log_prob = sum([true_like(observation[i]) for i in range(len(observation))])

    # Plot
    sns.scatterplot(x=observation[:, 0], y=observation[:, 1])
    plt.xlabel(r'$$x_1$$')
    plt.ylabel(r'$$x_2$$')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)

    plt.figure(dpi=200)
    plt.plot([2, 4], [-1.5, -1.5], color='k')
    plt.plot([3, 3], [-0.5, -2.5], color='k')
    plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin='lower')
    plt.axis('scaled')
    plt.xlim(2+0.3, 4-0.3)
    plt.ylim(-2.5+0.3, -0.5-0.3)
    plt.title(r'Posterior with learned likelihood\nfrom %d examples of'%(num_sim)+r' $$\mu_i\in[-5, 5]$$')
    plt.xlabel(r'$$\mu_1$$')
    plt.ylabel(r'$$\mu_2$$')

    plt.figure(dpi=200)
    prob = torch.exp(true_log_prob - true_log_prob.max())
    plt.plot([2, 4], [-1.5, -1.5], color='k')
    plt.plot([3, 3], [-0.5, -2.5], color='k')
    plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin='lower')
    plt.axis('scaled')
    plt.xlim(2+0.3, 4-0.3)
    plt.ylim(-2.5+0.3, -0.5-0.3)
    plt.title('Posterior with\nanalytic likelihood')
    plt.xlabel(r'$$\mu_1$$')
    plt.ylabel(r'$$\mu_2$$')

    plt.show()

