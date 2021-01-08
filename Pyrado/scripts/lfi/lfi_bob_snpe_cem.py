import sbi.utils as utils
import torch.nn as nn
from sbi.inference import SNPE, SNPE_B, SNPE_C, SNLE, prepare_for_sbi

import pyrado
from scripts.lfi.plot_trajectories import plot_trajectories
from scripts.lfi.train_lfi import *
from scripts.lfi.simulators import BallOnBeam


if __name__ == "__main__":
    num_sim = 5
    num_rounds = 10
    num_samples = 200
    n_observations = 5

    # define simulator
    simulator = BallOnBeam()
    prior = utils.BoxUniform(low=to.tensor([9.75, 0.2]), high=to.tensor([9.9, 0.8]))

    # real environment
    true_params = to.tensor([9.81, 0.5]).to(dtype=to.float32)
    x_o = simulator(true_params)

    # define algorithm hyperparameters
    input_length = len(x_o)
    embedding_net = nn.Linear(input_length, 10).to(dtype=to.float32)
    simulator, prior = prepare_for_sbi(simulator, prior)
    model = utils.posterior_nn(model="maf", hidden_features=10, num_transforms=2, embedding_net=embedding_net)

    inference = SNPE(prior, density_estimator=model)

    # train the posterior
    posterior = train_lfi(
        simulator, inference, prior, x_o, num_sim=num_sim, num_rounds=num_rounds, num_samples=num_samples
    )

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
