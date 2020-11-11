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
import torch as to
from matplotlib import pyplot as plt
from sbi.inference.base import infer

import pyrado
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout


plt.rcParams.update({"text.usetex": True})


def simulator(mu):
    # In the end, the output of this could be a distance measure over trajectories instead of just the final state
    ro = rollout(env, policy, eval=True, reset_kwargs=dict(
        # domain_param=dict(k=mu[0], d=mu[1]), init_state=np.array([-0.7, 0.])  # no variance over the init state
        domain_param=dict(k=mu[0], d=mu[1])  # no variance over the parameters
    ))
    return to.from_numpy(ro.observations[-1]).to(dtype=to.float32)


if __name__ == '__main__':
    pyrado.set_seed(0)

    env = OneMassOscillatorSim(dt=0.005, max_steps=200)
    policy = IdlePolicy(env.spec)

    prior = utils.BoxUniform(
        low=to.tensor([25., 0.05]),
        high=to.tensor([35., 0.15])
    )

    # Let’s learn a likelihood from the simulator
    num_sim = 500
    method = 'SNRE'  # SNPE or SNLE or SNRE
    posterior = infer(
        simulator,
        prior,
        method=method,  # SNRE newer than SNLE newer than SNPE
        num_workers=-1,
        num_simulations=num_sim)

    # Let’s record our “observations” of the true distribution
    n_observations = 5
    # noisy_true_params = to.tensor([30, 0.1]) + to.tensor([30, 0.1])*to.randn(n_observations, 2)/10  # no variance over the init state
    noisy_true_params = to.tensor([30, 0.1]).repeat((n_observations, 1))  # no variance over the parameters
    observation = to.stack([simulator(dp) for dp in noisy_true_params])

    # Inference
    # samples = posterior.sample((200,), x=observation[0])  # sample the posterior for a single data point
    samples = to.cat([posterior.sample((200,), x=obs) for obs in observation], dim=0)

    # Computing the log-probability
    bounds = [20, 40, 0.0, 0.2]
    mu_1, mu_2 = to.tensor(np.mgrid[bounds[0]:bounds[1]:20/50., bounds[2]:bounds[3]:0.2/50.]).float()
    grids = to.cat(
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
            posterior.net(to.cat([grids, observation[i].repeat((grids.shape[0], 1))], dim=1))[:, 0] +
            posterior._prior.log_prob(grids)
            for i in range(len(observation))
        ]).detach()
    prob = to.exp(log_prob - log_prob.max())  # scale the probabilities to [0, 1]

    # log_probability = posterior.log_prob(samples, x=observation[0])  # log_probability seems to be unused

    # Plot
    sns.scatterplot(x=observation[:, 0], y=observation[:, 1])
    plt.xlabel(r'$x_0$')
    plt.ylabel(r'$x_1$')

    out = utils.pairplot(samples, limits=[[20., 40.], [0.0, 0.2]], fig_size=(6, 6), upper='kde', diag='kde')

    plt.figure(dpi=200)
    plt.plot([20., 40.], [0.1, 0.1], color='w', ls='--')
    plt.plot([30., 30.], [0.0, 0.2], color='w', ls='--')
    plt.contourf(mu_1.numpy(), mu_2.numpy(), prob.reshape(*mu_1.shape).numpy())
    # plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin='lower')
    plt.title(f'Posterior with learned likelihood \n from {num_sim} examples of the true distribution')
    plt.xlabel(r'$k$')
    plt.ylabel(r'$d$')
    plt.show()
