import pyrado
import torch as to
import torch.nn as nn
from scripts.lfi.plot_thetas import plot_2d_thetas
from torch.distributions.uniform import Uniform
from torch.distributions.multivariate_normal import MultivariateNormal

from pyrado.logger.experiment import setup_experiment, ask_for_experiment
from pyrado.algorithms.inference.lfi2 import LFI
from pyrado.utils.argparser import get_argparser

from sbi.inference import SNPE
import sbi.utils as utils

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def toy_simulator(theta):
    """
    See SNL[1] Toy example

    [1] Papamakarios, George, David Sterratt, and Iain Murray.
    "Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows."
    The 22nd International Conference on Artificial Intelligence and Statistics. PMLR, 2019.
    """
    # if not (theta.shape == to.Size([1, 5]) or len(theta) == 5):
    #     raise pyrado.ShapeErr(given=theta)
    with to.no_grad():
        mean = theta[:2]
        s1 = theta[2] ** 2
        s2 = theta[3] ** 2
        rho = to.tanh(theta[4])
        cov12 = rho * s1 * s2
        covariance_matrix = to.tensor([[s1 ** 2, cov12], [cov12, s2 ** 2]])
        dist = MultivariateNormal(loc=mean, covariance_matrix=covariance_matrix)
        x = dist.sample((10,)).squeeze().mean(dim=0)

        return dist.sample((4,)).view(-1, 1).squeeze()


def create_setup():
    # parameter which LFI trains for
    simulator = toy_simulator

    # define prior and true parameter distributions
    range = 3 * to.ones((6,))
    prior = utils.BoxUniform(low=-range, high=range)
    # real_params = to.tensor([0.7, -2.9, -1, -0.9, 0.6])
    real_params = to.tensor([0.7, -2.9, -0.5, -0.5, 0.1])
    return simulator, prior, real_params


def create_sbi_algo():
    # Subroutine
    inference_hparam = dict(max_iter=5, num_sim=1000)
    embedding_net = nn.Identity()
    flow = utils.posterior_nn(model="maf", hidden_features=10, num_transforms=2, embedding_net=embedding_net)
    return flow, inference_hparam


if __name__ == "__main__":
    # Parse command line arguments
    parser = get_argparser()
    # choose --eval if you just want to evaluate a model
    parser.add_argument("--eval", action="store_true", default=False, help="evaluate the function")
    args = parser.parse_args()

    # Set the seed
    pyrado.set_seed(1001, verbose=True)

    # define RolloutSamplerForSBIBase
    simulator, prior, real_params = create_setup()
    num_obs = 1
    ro_real = [simulator(real_params) for _ in range(num_obs)]
    ro_real = to.stack(ro_real)

    num_samples = 100

    # create an experiment
    algo_name = "SNPE"
    if not args.eval:
        ex_dir = setup_experiment("toy_example", f"{algo_name}")
    else:
        ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # setup
    # define normalizing flow
    flow, inference_hparam = create_sbi_algo()
    # instantiate inference Algorithm
    inference = LFI(
        save_dir=ex_dir,
        simulator=simulator,
        flow=flow,
        inference=SNPE,
        prior=prior,
        params_names=None,
        **inference_hparam,
    )

    if not args.eval:
        # train the LFI algorithm
        inference.step(snapshot_mode="latest", meta_info=dict(rollouts_real=ro_real), logging=True)

        # load trajectories from training
        log_probs = pyrado.load(None, "log_probs", "pkl", ex_dir)
        n_simulations = pyrado.load(None, "n_simulations", "pkl", ex_dir)

        # plot log probabilities against n_simulations
        plt.figure()
        plt.plot(n_simulations, log_probs)
        plt.show()
    else:
        # load a saved posterior for inference instead of training it
        posterior = pyrado.load(None, "posterior", "pt", ex_dir)

        # update posterior in inference
        inference.set_posterior(posterior)

    # sample parameters
    sample_params, _, sim_rollouts = inference.evaluate(
        rollouts_real=ro_real, num_samples=num_samples, compute_quantity={"sample_params": True, "sim_traj": True}
    )

    sim_rollouts = sim_rollouts.unsqueeze(-1).view(num_obs, num_samples, 4, 2)
    ro_real = ro_real.view(num_obs, 4, 2)

    fig, ax = plt.subplots()
    colors = list(mcolors.TABLEAU_COLORS)  # list of color keys
    legend_elements = [Line2D([0], [0], marker="o", color="black", label="Sampled Observations", markersize=10)]

    # plot samples from proposals
    for cnt, ro in enumerate(sim_rollouts):
        for sample in sample_params:
            ax.scatter(sample[:, 0], sample[:, 1], c=colors[cnt], label="Samples", alpha=0.3)

    for cnt, ro in enumerate(ro_real):
        for sample in ro:
            ax.scatter(sample[0], sample[1], facecolors=colors[cnt], marker="X", edgecolors="black", s=200)

    plt.show()
