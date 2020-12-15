from collections import Callable
import torch as to
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import simulate_for_sbi
from sbi.inference.posteriors.direct_posterior import DirectPosterior


def train_lfi(simulator,
              inference,
              prior,
              x_o,
              num_rounds: int = 5,
              num_sim: int = 1000,
              summary: SummaryWriter = None
              ):
    proposal_prior = prior
    for _ in range(num_rounds):
        theta, x = simulate_for_sbi(simulator, proposal_prior, num_simulations=num_sim, simulation_batch_size=1)
        _ = inference.append_simulations(theta, x).train()
        proposal_prior = inference.build_posterior().set_default_x(x_o)
    posterior = proposal_prior
    return posterior


def evaluate_lfi(simulator: Callable,
                 posterior: DirectPosterior,
                 x_o,
                 num_samples: int = 1000):

    proposals = posterior.sample((num_samples, ), x=x_o)
    log_prob = posterior.log_prob(proposals, x=x_o)
    trajectories = []
    for proposal in proposals:
        trajectory = simulator(proposal.unsqueeze(0))
        trajectories.append(trajectory)
    return proposals, log_prob, trajectories


def save_nn(nn: to.nn.Module, path: str):
    to.save(nn.state_dict(), path)
    print("saved model at:\t {}".format(path))


def load_nn(nn, path: str):
    state_dict = to.load(path)
    nn.load_state_dict(state_dict=state_dict)
    print("loaded model from: {}".format(path))
