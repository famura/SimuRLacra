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
Testing the simulation-based inference (SBI) toolbox using a very basic example
"""
import functools
import numpy as np
import sbi.utils as utils
import torch as to
from matplotlib import pyplot as plt
from sbi.inference.base import infer

import pyrado
from pyrado.algorithms.inference.sbi_rollout_sampler import SimRolloutSamplerForSBI
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.environments.sim_base import SimEnv
from pyrado.plotting.distribution import draw_posterior_distr_2d
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.policies.base import Policy
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout
from pyrado.spaces.singular import SingularStateSpace


def simple_omo_sim(mu: to.Tensor, env: SimEnv, policy: Policy) -> to.Tensor:
    """ The most simple interface of a simulation to sbi, see `SimRolloutSamplerForSBI` """
    ro = rollout(env, policy, eval=True, reset_kwargs=dict(domain_param=dict(k=mu[0], d=mu[1])))
    observation_sim = to.from_numpy(ro.observations[-1]).to(dtype=to.float32)
    return to.atleast_2d(observation_sim)


if __name__ == "__main__":
    # Config
    plt.rcParams.update({"text.usetex": True})
    basic_wrapper = False
    pyrado.set_seed(0)

    # Environment and policy
    env = OneMassOscillatorSim(dt=1 / 200, max_steps=200)
    env.init_space = SingularStateSpace(np.array([-0.7, 0]))  # no variance over the initial state
    policy = IdlePolicy(env.spec)

    # Domain parameter mapping and prior, oly use 2 domain parameters here to simplify the plotting later
    dp_mapping = {0: "k", 1: "d"}
    prior = utils.BoxUniform(low=to.tensor([20.0, 0.0]), high=to.tensor([40.0, 0.3]))

    # Create the simulator compatible with sbi
    if basic_wrapper:
        simulator = functools.partial(simple_omo_sim, env=env, policy=policy)
    else:
        simulator = SimRolloutSamplerForSBI(env, policy, dp_mapping, strategy="final_state", num_segments=1)

    # Learn a likelihood from the simulator
    num_sim = 500
    method = "SNPE"
    posterior = infer(simulator, prior, method=method, num_simulations=num_sim, num_workers=1)

    # Create a fake (noisy) true distribution
    num_observations = 6
    dp_gt = {"k": 30, "d": 0.1}
    domain_param_gt = to.tensor([dp_gt[key] for _, key in dp_mapping.items()])
    domain_param_gt = domain_param_gt.repeat((num_observations, 1))
    domain_param_gt += domain_param_gt * to.randn(num_observations, 2) / 5
    observations_real = to.cat([simulator(dp) for dp in domain_param_gt], dim=0)

    # Plot the posterior
    _, axs = plt.subplots(*num_rows_cols_from_length(num_observations), figsize=(14, 14), tight_layout=True)
    draw_posterior_distr_2d(
        axs, "separate", posterior, observations_real, dp_mapping, dims=(0, 1), condition=to.zeros(2), prior=prior
    )

    # Plot the ground truth domain parameters
    for idx, dp_gt in enumerate(domain_param_gt):
        axs[idx // axs.shape[1], idx % axs.shape[1]].scatter(
            x=dp_gt[0], y=dp_gt[1], marker="o", s=30, zorder=3, color="white", edgecolors="black"
        )

    plt.show()
