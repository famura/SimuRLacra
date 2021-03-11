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
from sbi.inference import simulate_for_sbi, SNPE_C
from sbi.user_input.user_input_checks import prepare_for_sbi
from sbi.utils import posterior_nn

import pyrado
from pyrado.algorithms.inference.embeddings import Embedding, LastStepEmbedding
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.environments.sim_base import SimEnv
from pyrado.plotting.distribution import draw_posterior_distr_2d
from pyrado.plotting.utils import num_rows_cols_from_length
from pyrado.policies.base import Policy
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout
from pyrado.spaces.singular import SingularStateSpace


def simple_omo_sim(domain_params: to.Tensor, env: SimEnv, policy: Policy) -> to.Tensor:
    """ The most simple interface of a simulation to sbi, see `SimRolloutSamplerForSBI` """
    domain_params = to.atleast_2d(domain_params)
    data = []
    for dp in domain_params:
        ro = rollout(
            env,
            policy,
            eval=True,
            stop_on_done=False,
            reset_kwargs=dict(domain_param=dict(k=dp[0], d=dp[1])),
        )
        data.append(to.from_numpy(ro.observations).to(dtype=to.get_default_dtype()))
    data = to.stack(data, dim=0).unsqueeze(1)  # batched domain param int the 1st dim and one rollout in the 2nd dim
    return Embedding.pack(data)


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

    # Wrap the simulator to abstract the env and the policy away from sbi
    w_simulator = functools.partial(simple_omo_sim, env=env, policy=policy)

    # Learn a likelihood from the simulator
    embedding = LastStepEmbedding(env.spec, dim_data=env.spec.obs_space.flat_dim)
    density_estimator = posterior_nn(model="maf", embedding_net=embedding, hidden_features=20, num_transforms=4)
    snpe = SNPE_C(prior, density_estimator)
    simulator, prior = prepare_for_sbi(w_simulator, prior)
    domain_param, data_sim = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=50,
        num_workers=1,
    )
    snpe.append_simulations(domain_param, data_sim)
    density_estimator = snpe.train()
    posterior = snpe.build_posterior(density_estimator)

    # Create a fake (random) true distribution
    num_instances_real = 1
    dp_gt = {"k": 30, "d": 0.1}
    domain_param_gt = to.tensor([dp_gt[key] for _, key in dp_mapping.items()])
    domain_param_gt = domain_param_gt.repeat((num_instances_real, 1))
    domain_param_gt += domain_param_gt * to.randn(num_instances_real, 2) / 5
    data_real = to.cat([simulator(dp) for dp in domain_param_gt], dim=0)
    data_real = Embedding.unpack(data_real, dim_data_orig=env.spec.obs_space.flat_dim)

    # Plot the posterior
    _, axs = plt.subplots(*num_rows_cols_from_length(num_instances_real), figsize=(14, 14), tight_layout=True)
    axs = np.atleast_2d(axs)
    draw_posterior_distr_2d(
        axs, "separate", posterior, data_real, dp_mapping, dims=(0, 1), condition=to.zeros(2), prior=prior
    )

    # Plot the ground truth domain parameters
    for idx, dp_gt in enumerate(domain_param_gt):
        axs[idx // axs.shape[1], idx % axs.shape[1]].scatter(
            x=dp_gt[0], y=dp_gt[1], marker="o", s=30, zorder=3, color="white", edgecolors="black"
        )

    plt.show()
