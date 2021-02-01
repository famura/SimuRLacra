"""
Sim-to-sim experiment on the Quanser Qube environment using likelihood-free inference
"""
import torch as to
import torch.nn as nn
from copy import deepcopy
from sbi.inference import SNPE
from sbi import utils

import pyrado
from pyrado.algorithms.inference.lfi import LFI
from pyrado.domain_randomization.domain_parameter import NormalDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.utils.argparser import get_argparser

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{LFI.name}_{QQubeSwingUpAndBalanceCtrl.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_hparams = dict(dt=1 / 250.0, max_steps=1500)
    env_sim = QQubeSwingUpSim(**env_hparams)

    # Create a fake ground truth target domain
    num_real_obs = 1
    env_real = deepcopy(env_sim)
    # randomizer = DomainRandomizer(
    #     NormalDomainParam(name="g", mean=10.0, std=10.0 / 20),
    #     NormalDomainParam(name="Rm", mean=9.0, std=9.0 / 20),
    #     NormalDomainParam(name="Mp", mean=0.02, std=0.02 / 20),
    # )
    # env_real = DomainRandWrapperBuffer(env_real, randomizer)
    # env_real.fill_buffer(num_real_obs)
    # dp_mapping = {0: "Dr", 1: "Dp", 2: "Rm", 3: "km"}
    dp_mapping = {0: "Dr", 1: "Dp", 2: "Rm", 3: "km", 4: "Mr", 5: "Mp", 6: "Lr", 7: "Lp"}
    # Policy
    behavior_policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec)

    # Prior and Posterior (normalizing flow)
    dp_nom = env_sim.get_nominal_domain_param()
    prior_hparam = dict(
        # low=to.tensor([dp_nom["Dr"] * 0, dp_nom["Dp"] * 0, dp_nom["Rm"] * 0.5, dp_nom["km"] * 0.5]),
        # high=to.tensor([dp_nom["Dr"] * 10, dp_nom["Dp"] * 10, dp_nom["Rm"] * 1.5, dp_nom["km"] * 1.5]),
        low=to.tensor(
            [
                dp_nom["Dr"] * 0,
                dp_nom["Dp"] * 0,
                dp_nom["Rm"] * 0.8,
                dp_nom["km"] * 0.8,
                dp_nom["Mr"] * 0.9,
                dp_nom["Mp"] * 0.9,
                dp_nom["Lr"] * 0.9,
                dp_nom["Lp"] * 0.9,
            ]
        ),
        high=to.tensor(
            [
                dp_nom["Dr"] * 10,
                dp_nom["Dp"] * 10,
                dp_nom["Rm"] * 1.2,
                dp_nom["km"] * 1.2,
                dp_nom["Mr"] * 1.1,
                dp_nom["Mp"] * 1.1,
                dp_nom["Lr"] * 1.1,
                dp_nom["Lp"] * 1.1,
            ]
        ),
    )
    prior = utils.BoxUniform(**prior_hparam)
    posterior_nn_hparam = dict(model="maf", embedding_net=nn.Identity(), hidden_features=50, num_transforms=5)

    # Algorithm
    algo_hparam = dict(
        summary_statistic="bayessim",
        max_iter=5,
        num_real_rollouts=num_real_obs,
        num_sim_per_real_rollout=4000,
        simulation_batch_size=1,
        use_posterior_in_the_loop=False,
        normalize_posterior=False,
        num_workers=10,
    )
    algo = LFI(
        ex_dir,
        env_sim,
        env_real,
        behavior_policy,
        dp_mapping,
        prior,
        posterior_nn_hparam,
        SNPE,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparams, seed=args.seed),
        dict(prior=prior_hparam),
        dict(posterior_nn=posterior_nn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    algo.train(seed=args.seed)
