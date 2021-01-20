import pyrado
import torch as to
import torch.nn as nn
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim

from pyrado.logger.experiment import setup_experiment, ask_for_experiment
from pyrado.algorithms.inference.lfi import LFI
from pyrado.algorithms.inference.sbi_rollout_sampler import EnvSimulator
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import RenderMode
from scripts.lfi.plot_thetas import plot_2d_thetas
from scripts.lfi.normalizing_flows import train_nflows

from sbi.inference import SNPE
import sbi.utils as utils


def render_sim(env, policy):
    rollout(env, policy, eval=True, render_mode=RenderMode(video=True))


def create_qq_setup():
    env_hparams = dict(dt=1 / 100.0, max_steps=3500)
    env_sim = QQubeSwingUpSim(**env_hparams)
    behavior_policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)
    # render_sim(env_sim, behavior_policy)

    # parameter which LFI trains for
    params_names = list(env_sim.get_nominal_domain_param().keys())
    simulator = EnvSimulator(env_sim, behavior_policy, params_names, "ramos")

    # define prior and true parameter distributions
    real_params = to.tensor(list(env_sim.get_nominal_domain_param().values()), dtype=to.float32)
    prior = utils.BoxUniform(low=0.9 * real_params, high=1.1 * real_params)
    return simulator, prior, real_params


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

    # define RolloutSamplerForSBI
    simulator, prior, real_params = create_qq_setup()

    # sample from true parameter distribution and generate observations with it
    num_obs = 1
    ro_real = [simulator(real_params) for _ in range(num_obs)]

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
        **inference_hparam,
    )

    if not args.eval:
        # train the LFI algorithm
        inference.step(snapshot_mode="latest", meta_info=dict(rollouts_real=ro_real))
    else:
        # load a saved posterior for inference instead of training it
        posterior = pyrado.load(None, "posterior", "pt", ex_dir)

        # update posterior in inference
        inference.set_posterior(posterior)

    sample_params, _, _ = inference.evaluate(
        meta_info=dict(rollouts_real=ro_real), num_samples=num_samples, compute_quantity={"sample_params": True}
    )

    print(sample_params[0].shape)
