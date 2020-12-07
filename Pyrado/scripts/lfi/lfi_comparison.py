"""
...

"""
import numpy as np
import sbi.utils as utils
import seaborn as sns
import torch as to
import torch.nn as nn
from matplotlib import pyplot as plt
from sbi.inference.base import infer
from sbi.inference import SNPE, SNPE_B, SNPE_C, SNLE, prepare_for_sbi, simulate_for_sbi

import pyrado
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.sampling.rollout import rollout


def fancy_simulator(mu):
    # In the end, the output of this could be a distance measure over trajectories instead of just the final state
    ro = rollout(env, policy, eval=True, reset_kwargs=dict(
        domain_param=dict(k=mu[0], d=mu[1])
    ))
    return to.tensor(ro.observations).to(dtype=to.float32).view(-1, 1).squeeze()


if __name__ == '__main__':
    num_sim = 5
    num_rounds = 1
    num_samples = 200
    n_observations = 5
    name = "SNPE"

    # define simulator
    env = OneMassOscillatorSim(dt=0.005, max_steps=200)
    policy = IdlePolicy(env.spec)
    simulator = fancy_simulator
    true_params = to.tensor([30, 0.1])
    x_o = simulator(true_params)

    # define proposal prior, algorithm and more

    prior = utils.BoxUniform(
        low=to.tensor([25., 0.05]),
        high=to.tensor([35., 0.15])
    )

    input_length = len(x_o)
    embedding_net = nn.Linear(input_length, 10).to(dtype=to.float32)
    simulator, prior = prepare_for_sbi(simulator, prior)

    if "SNPE" in name:
        model = utils.posterior_nn(model='maf', hidden_features=10,
                                   num_transforms=2, embedding_net=embedding_net)
    elif "SNLE":
        model = utils.likelihood_nn(model='maf', hidden_features=10,
                                              num_transforms=2)
    else:
        model = None

    inference = {"SNPE-B": SNPE_B,
                "SNPE-C": SNPE_C,
                "SNPE": SNPE,
                "SNLE": SNLE
                }[name](prior, density_estimator=model)

    # training function
    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, prior, num_simulations=num_sim, simulation_batch_size=1)
        _ = inference.append_simulations(theta, x).train()
        prior = inference.build_posterior().set_default_x(x_o)
    posterior = prior

