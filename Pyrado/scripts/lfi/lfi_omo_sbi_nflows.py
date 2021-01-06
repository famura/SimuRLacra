import numpy as np
import sbi.utils as utils
import seaborn as sns
import torch as to
import torch.nn as nn
from matplotlib import pyplot as plt
from sbi.inference.base import infer
from sbi.inference import SNPE, SNPE_B, SNPE_C, SNLE, prepare_for_sbi, simulate_for_sbi

import pyrado
from scripts.lfi.plot_trajectories import plot_trajectories
from scripts.lfi.plot_thetas import plot_2d_thetas
from scripts.lfi.normalizing_flows import train_nflows
from scripts.lfi.train_lfi import *
from scripts.lfi.simulators import OscillatorTrajectories


if __name__ == "__main__":
    num_sim = 200  # 100
    num_rounds = 5  # 5
    num_samples = 1000  # 1000
    n_observations = 1  # 1
    nflows_iter = 5000

    # define simulator
    simulator = OscillatorTrajectories()
    prior = utils.BoxUniform(low=to.tensor([15, 0.1]), high=to.tensor([40.0, 0.3]))

    def normal_dist(n):
        observation_prior = utils.BoxUniform(low=to.tensor([25.0, 0.15]), high=to.tensor([35.0, 0.25]))
        return observation_prior.sample((n,))

    # ---- hand crafted observation_thetas
    # circle shape
    def circle_dist(n):
        theta_1 = to.rand(n) * 10 + 25  # values between 25 and 35
        theta_2 = (5**2 - (theta_1 - 30)**2)**(1/2) * 0.01 + 0.2
        return to.stack([theta_1, theta_2], 1)

    obs_theta = circle_dist(n_observations)

    x_o = [simulator(param) for param in obs_theta]
    x_o = to.stack(x_o)

    # generate more observations for evaluation
    obs_theta_eval = circle_dist(10)
    x_o_eval = to.stack([simulator(param) for param in obs_theta_eval])

    # x_o = x_o.unsqueeze(0)  # add a second dim for stacking
    # for i in range(n_observations - 1):
    #     test_prior = utils.BoxUniform(low=to.tensor([15.0, 0.0]), high=to.tensor([55.0, 1.0]))
    #     x_o = to.cat([x_o, simulator(test_prior.sample().unsqueeze(0))], dim=0)

    # real environment
    # true_params = to.tensor([30, 0.1]).to(dtype=to.float32)
    # x_o = simulator(true_params)
    # x_o = [x_o for _ in range(5)]

    # define algorithm hyperparameters
    input_length = x_o.shape[1]
    # embedding_net = nn.Identity()
    embedding_net = nn.Linear(input_length, 100).to(dtype=to.float32)
    simulator, prior = prepare_for_sbi(simulator, prior)
    model = utils.posterior_nn(model="maf", hidden_features=10, num_transforms=2, embedding_net=embedding_net)

    inference = SNPE(prior, density_estimator=model)

    # train the posterior
    posterior = train_lfi(simulator, inference, prior, x_o, num_sim=num_sim, num_rounds=num_rounds)

    # generate more observations
    '''
    x_o = x_o.unsqueeze(0)  # add a second dim for stacking
    for i in range(n_observations - 1):
        test_prior = utils.BoxUniform(low=to.tensor([15.0, 0.0]), high=to.tensor([55.0, 1.0]))
        x_o = to.cat([x_o, simulator(test_prior.sample().unsqueeze(0))], dim=0)
    '''

    # generate examples from the posterior
    proposals, log_prob, trajectories = evaluate_lfi(
        simulator=simulator, posterior=posterior, observations=x_o_eval, num_samples=num_samples
    )

    # save model
    save_nn(posterior.net, "model_snpe")

    # testing to load the model
    # new_model = utils.posterior_nn(model='maf', hidden_features=10,
    #                                num_transforms=2, embedding_net=embedding_net)
    # new_inference = SNPE(prior, density_estimator=new_model)
    # new_posterior = train_lfi(simulator, inference, prior, x_o, num_sim=1, num_rounds=1)
    # load_nn(new_posterior.net, "sbi_logs")

    # TODO: come up with smarter way to load to initialize the posterior
    # new_posterior = DirectPosterior(method_family="snpe",
    #                                 neural_net=model(true_params.unsqueeze(0), x_o.unsqueeze(0)),
    #                                 prior=prior,
    #                                 x_shape=x_o.shape)

    # sample from marginal posterior
    def sample_from_marginal(proposals, s_num=100):
        # https://projecteuclid.org/download/pdfview_1/euclid.ba/1459772735
        m_sample = None
        # Arithmetic Mean Estimator
        def AME(prop):
            return to.mean(prop, 0)
        # Harmonic Mean Estimator
        def HME(prop):
            return 1 / ((1 / prop.shape[0]) * to.sum(1 / prop, 0))

        method = HME # select a method for marginalization

        return to.stack([method(proposals[:, s, :]) for s in range(s_num)], dim=0)


    # marginals = sample_from_marginal(proposals, s_num=num_samples)
    marginals = None

    prior = train_nflows(to.cat([obs for obs in proposals], 0), num_iter=nflows_iter)

    marginals = prior.sample(num_samples).detach()
    # plot useful statistics
    plot_2d_thetas(proposals[:9, :, :], obs_thetas=obs_theta_eval[:9, :], marginal_samples=marginals)

    # plot only nflow draws and original observations
    plot_2d_thetas(marginals.unsqueeze(0), obs_thetas=obs_theta_eval[:9, :], marginal_samples=None)

    # plot_trajectories(trajectories, n_parameter=2, observation_data=x_o)
