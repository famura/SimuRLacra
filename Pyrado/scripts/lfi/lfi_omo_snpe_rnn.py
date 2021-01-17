import sbi.utils as utils
import torch.nn as nn
from sbi.inference import SNPE, SNPE_B, SNPE_C, SNLE, prepare_for_sbi, simulate_for_sbi

import pyrado
from scripts.lfi.plot_trajectories import plot_trajectories
from scripts.lfi.train_lfi import *
from scripts.lfi.simulators import OscillatorTrajectories


if __name__ == "__main__":
    num_sim = 1000
    num_rounds = 3
    num_samples = 100
    n_observations = 5

    # define simulator
    simulator = OscillatorTrajectories(out_concat=False)
    prior = utils.BoxUniform(low=to.tensor([25.0, 0.05]), high=to.tensor([35.0, 0.15]))

    # real environment
    true_params = to.tensor([30, 0.1]).to(dtype=to.float32)
    x_o = simulator(true_params)

    # define algorithm hyperparameters
    input_length = len(x_o)
    embedding_net = nn.RNN(input_size=input_length, hidden_size=10, num_layers=2).to(dtype=to.float32)
    simulator, prior = prepare_for_sbi(simulator, prior)
    model = utils.posterior_nn(model="maf", hidden_features=10, num_transforms=2, embedding_net=embedding_net)

    inference = SNPE(prior, density_estimator=model)

    # train the posterior
    posterior = train_lfi(simulator, inference, prior, x_o, num_sim=num_sim, num_rounds=num_rounds)

    # generate more observations
    x_o = x_o.unsqueeze(0)  # add a second dim for stacking
    for i in range(n_observations - 1):
        test_prior = utils.BoxUniform(low=to.tensor([25.0, 0.0]), high=to.tensor([35.0, 1.0]))
        x_o = to.cat([x_o, simulator(test_prior.sample().unsqueeze(0))], dim=0)

    # generate examples from the posterior
    proposals, log_prob, trajectories = evaluate_lfi(
        simulator=simulator, posterior=posterior, observations=x_o, num_samples=num_samples
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

    # plot useful statistics

    plot_trajectories(trajectories, n_parameter=2, observation_data=x_o)
