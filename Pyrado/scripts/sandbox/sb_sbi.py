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
import torch.nn as nn
from matplotlib import pyplot as plt
from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi

import pyrado
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout

# test

plt.rcParams.update({"text.usetex": False})


def simulator(mu):
    # In the end, the output of this could be a distance measure over trajectories instead of just the final state
    ro = rollout(
        env,
        policy,
        eval=True,
        reset_kwargs=dict(
            # domain_param=dict(k=mu[0], d=mu[1]), init_state=np.array([-0.7, 0.])  # no variance over the init state
            domain_param=dict(k=mu[0], d=mu[1])  # no variance over the parameters
        ),
    )
    return to.from_numpy(ro.observations[-1]).to(dtype=to.float32)


def simulator2(mu):
    # In the end, the output of this could be a distance measure over trajectories instead of just the final state
    ro = rollout(
        env,
        policy,
        eval=True,
        reset_kwargs=dict(
            # domain_param=dict(k=mu[0], d=mu[1]), init_state=np.array([-0.7, 0.])  # no variance over the init state
            domain_param=dict(k=mu[0], d=mu[1])  # no variance over the parameters
        ),
    )
    return to.tensor(ro.observations).to(dtype=to.float32).view(-1, 1).squeeze()


if __name__ == "__main__":
    pyrado.set_seed(0)

    env = OneMassOscillatorSim(dt=0.005, max_steps=200)
    policy = IdlePolicy(env.spec)

    prior = utils.BoxUniform(low=to.tensor([25.0, 0.05]), high=to.tensor([35.0, 0.15]))

    input_space = 402
    # embedding_net = SummaryNet()
    embedding_net = nn.Linear(input_space, 10).to(dtype=to.float32)
    # Let’s learn a likelihood from the simulator
    num_sim = 5
    num_rounds = 1
    num_samples = 200
    n_observations = 5
    method = "SNPE"  # SNPE or SNLE or SNRE

    true_params = to.tensor([30, 0.1])
    x_o = simulator2(true_params)
    print(x_o.shape)

    simulator, prior = prepare_for_sbi(simulator2, prior)
    neural_posterior = utils.posterior_nn(
        model="maf", hidden_features=10, num_transforms=2, embedding_net=embedding_net
    )
    inference = SNPE(prior, density_estimator=neural_posterior)

    # inference = SNPE(prior)

    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_sim, simulation_batch_size=1)
        _ = inference.append_simulations(theta, x).train()
        posterior = inference.build_posterior().set_default_x(x_o)

    x_o = to.stack([x_o for _ in range(n_observations)])
    print(x_o.shape)
    samples = to.cat([posterior.sample((num_samples,), x=obs) for obs in x_o], dim=0)

    bounds = [20, 40, 0.0, 0.2]
    mu_1, mu_2 = to.tensor(np.mgrid[bounds[0] : bounds[1] : 20 / 50.0, bounds[2] : bounds[3] : 0.2 / 50.0]).float()
    grids = to.cat((mu_1.reshape(-1, 1), mu_2.reshape(-1, 1)), dim=1)
    if method == "SNPE":
        log_prob = sum([posterior.log_prob(grids, x_o[i]) for i in range(len(x_o))])
    else:
        log_prob = sum(
            [
                posterior.net(to.cat([grids, x_o[i].repeat((grids.shape[0], 1))], dim=1))[:, 0]
                + posterior._prior.log_prob(grids)
                for i in range(len(x_o))
            ]
        ).detach()
    prob = to.exp(log_prob - log_prob.max())  # scale the probabilities to [0, 1]

    # log_probability = posterior.log_prob(samples, x=observation[0])  # log_probability seems to be unused

    # Plot
    sns.scatterplot(x=x_o[:, 0], y=x_o[:, 1])
    plt.xlabel(r"$x_0$")
    plt.ylabel(r"$x_1$")

    out = utils.pairplot(samples, limits=[[20.0, 40.0], [0.0, 0.2]], fig_size=(6, 6), upper="kde", diag="kde")

    plt.figure(dpi=200)
    plt.plot([20.0, 40.0], [0.1, 0.1], color="w", ls="--")
    plt.plot([30.0, 30.0], [0.0, 0.2], color="w", ls="--")
    plt.contourf(mu_1.numpy(), mu_2.numpy(), prob.reshape(*mu_1.shape).numpy())
    # plt.contourf(prob.reshape(*mu_1.shape), extent=bounds, origin='lower')
    plt.title(f"Posterior with learned likelihood \n from {num_sim} examples of the true distribution")
    plt.xlabel(r"$k$")
    plt.ylabel(r"$d$")
    plt.show()
