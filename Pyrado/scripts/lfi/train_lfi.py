from collections import Callable
import torch as to
from torch.utils.tensorboard import SummaryWriter

from sbi.inference import simulate_for_sbi
from sbi.inference.posteriors.direct_posterior import DirectPosterior
import log_prob_plot

# Test
def train_lfi(
    simulator,
    inference,
    prior,
    x_o,
    num_rounds: int = 5,
    num_sim: int = 1000,
    summary: SummaryWriter = None,
    eval_plot: bool = True,
):
    proposal_prior = prior
    if x_o.ndim == 1:
        for _ in range(num_rounds):
            theta, x = simulate_for_sbi(simulator, proposal_prior, num_simulations=num_sim, simulation_batch_size=1)
            _ = inference.append_simulations(theta, x).train()
            proposal_prior = inference.build_posterior().set_default_x(x_o)
        posterior = proposal_prior
    else:
        theta, x = simulate_for_sbi(simulator, proposal_prior, num_simulations=num_sim, simulation_batch_size=1)
        _ = inference.append_simulations(theta, x).train()
        proposal_prior = inference.build_posterior()
        if num_rounds > 1:
            for _ in range(num_rounds - 1):
                for obs in x_o:
                    proposal_prior.set_default_x(obs)
                    theta, x = simulate_for_sbi(simulator, proposal_prior, num_simulations=num_sim,
                                                simulation_batch_size=1)
                    _ = inference.append_simulations(theta, x)
                _ = inference.train()
                proposal_prior = inference.build_posterior()
            posterior = proposal_prior
        else:
            posterior = proposal_prior
    return posterior


def evaluate_lfi(simulator: Callable, posterior: DirectPosterior, observations,
                 num_samples: int = 1000, comp_log_prob=False, comp_trajectory=False):
    """
        INPUT:
    ...
    observations:   to.Tensor[num_observations, trajectory_size]

        OUTPUT:
    proposals:      to.Tensor[num_observations, num_samples, parameter_size]
    log_prob:       to.Tensor[?, ?]
    trajectories:   to.Tensor[num_observations, num_samples, trajectory_size]
    """
    proposals, log_prob, trajectories = None, None, None
    num_observations = observations.shape[0]
    trajectory_size = observations.shape[1]

    # compute proposals for each observation
    proposals = to.stack([posterior.sample((num_samples,), x=obs) for obs in observations], dim=0)

    trajectories = to.empty((num_observations, num_samples, trajectory_size))
    log_prob = to.empty((num_observations, num_samples))
    cnt = 0
    for o in range(num_observations):
        # compute log probability
        if comp_log_prob:
            log_prob[o, :] = posterior.log_prob(proposals[o, :, :], x=observations[o, :])
        if comp_trajectory:
            for s in range(num_samples):
                if not s % 10:
                    print(
                        "\r[train_lfi.py/evaluate_lfi] Observation: ({}|{}), Sample: ({}|{})".format(
                            o, num_observations, s, num_samples
                        ),
                        end="",
                    )
                # compute trajectories for each observation and every sample
                trajectories[o, s, :] = simulator(proposals[o, s, :].unsqueeze(0))
        cnt += 1

    return proposals, log_prob, trajectories


def save_nn(nn: to.nn.Module, path: str):
    to.save(nn.state_dict(), path)
    print("saved model at:\t {}".format(path))


def load_nn(nn, path: str):
    state_dict = to.load(path)
    nn.load_state_dict(state_dict=state_dict)
    print("loaded model from: {}".format(path))
