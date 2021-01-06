import pyrado
import torch as to
import torch.nn as nn

from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.environments.pysim.one_mass_oscillator import OneMassOscillatorSim
from pyrado.logger.experiment import setup_experiment, ask_for_experiment
from pyrado.algorithms.inference.lfi import LFI, EnvSimulator
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.utils.argparser import get_argparser
from scripts.lfi.plot_thetas import plot_2d_thetas
from scripts.lfi.normalizing_flows import train_nflows

from sbi.inference import SNPE
import sbi.utils as utils


def create_omo_setup():
    env_hparams = dict(dt=1 / 100.0, max_steps=200)
    env_sim = OneMassOscillatorSim(**env_hparams)
    behavior_policy = IdlePolicy(env_sim.spec)
    # parameter which LFI trains for
    params_names = ["k", "d"]
    simulator = EnvSimulator(env_sim, behavior_policy, params_names)

    # define prior and true parameter distributions
    prior = utils.BoxUniform(low=to.tensor([27.0, 0.05]), high=to.tensor([33, 0.15]))
    real_param_dist = to.distributions.MultivariateNormal(to.tensor([30, 0.1]), to.tensor([30, 0.1]) / 100 * to.eye(2))
    return simulator, prior, real_param_dist, params_names


def create_sbi_algo():
    # Subroutine
    inference_hparam = dict(max_iter=5, num_sim=10)
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

    # define Simulator
    simulator, prior, real_param_dist, params_names = create_omo_setup()

    # sample from true parameter distribution and generate observations with it
    num_real_obs = 5
    real_params = real_param_dist.sample((num_real_obs,))
    ro_real = [simulator(param) for param in real_params]
    ro_real = to.stack(ro_real)

    num_samples = 100

    # create an experiment
    algo_name = "SNPE"
    if not args.eval:
        ex_dir = setup_experiment(simulator.name, f"{algo_name}")
    else:
        ex_dir = ask_for_experiment() if args.dir is None else args.dir

    # setup
    # define normalizing flow
    flow, inference_hparam = create_sbi_algo()
    # instantiate inference Alogorithm
    inference = LFI(
        save_dir=ex_dir,
        simulator=simulator,
        flow=flow,
        inference=SNPE,
        prior=prior,
        params_names=params_names,
        **inference_hparam,
    )

    if not args.eval:
        # train the LFI algorithm
        inference.step(snapshot_mode="latest", meta_info=dict(rollouts_real=ro_real))
        sample_params, _, _ = inference.evaluate(
            obs_traj=ro_real, num_samples=num_samples, compute_quantity={"sample_params": True}
        )
    else:
        # load a saved posterior for inference instead of training it
        posterior = pyrado.load(None, "posterior", "pt", ex_dir)

        # update posterior in inference
        inference.set_posterior(posterior)

        # generate parameters
        sample_params, _, _ = inference.evaluate(
            obs_traj=ro_real, num_samples=num_samples, compute_quantity={"sample_params": True}
        )

    # estimate prior with normalizing flows
    nflows_iter=5000
    prior = train_nflows(to.cat([obs for obs in sample_params], 0), num_iter=nflows_iter)
    marginals = prior.sample(num_samples).detach()

    # plot useful statistics
    plot_2d_thetas(
        sample_params, obs_thetas=real_params, marginal_samples=marginals
    )
    # plot_trajectories(trajectories, n_parameter=2, observation_data=x_o)
