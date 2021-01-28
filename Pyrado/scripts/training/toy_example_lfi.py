from copy import deepcopy
import torch as to
import torch.nn as nn

import pyrado
from pyrado.algorithms.inference.lfi import LFI
from pyrado.domain_randomization.domain_parameter import NormalDomainParam, UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environments.one_step.multivariate_gaussian import ToyExample
from pyrado.logger.experiment import setup_experiment, save_dicts_to_yaml
from pyrado.policies.special.dummy import IdlePolicy
from pyrado.utils.argparser import get_argparser

import sbi.utils as utils
from sbi.inference import SNPE

if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(ToyExample.name, f"{LFI.name}")

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environments
    env_sim = ToyExample()

    test = env_sim.spec

    # Create a fake 'ground truth' target domain
    num_real_obs = 10
    env_real = deepcopy(env_sim)

    # # either run with the same observation every round or...
    # randomizer = DomainRandomizer(
    #     UniformDomainParam(name="m_1", mean=0.7, halfspan=1e-6),
    #     UniformDomainParam(name="m_2", mean=-1.5, halfspan=1e-6),
    #     UniformDomainParam(name="s_1", mean=-0.5, halfspan=1e-6),
    #     UniformDomainParam(name="s_2", mean=-0.5, halfspan=1e-6),
    #     UniformDomainParam(name="rho", mean=0.1, halfspan=1e-6),
    # )
    # env_real = DomainRandWrapperBuffer(env_real, randomizer)
    # env_real.fill_buffer(num_real_obs)

    # # ... sample every round a new observation
    env_real.domain_param = dict(m_1=0.7, m_2=-2.9, s_1=-1, s_2=-0.9, rho=0.6)
    # env_real.domain_param = dict(m_1=0.7, m_2=-1.5, s_1=-0.1, s_2=-0.1, rho=0.1)
    dp_mapping = {0: "m_1", 1: "m_2", 2: "s_1", 3: "s_2", 4: "rho"}

    # Policy
    behavior_policy = IdlePolicy(env_sim.spec)

    # Prior
    prior_hparam = dict(low=-3 * to.ones((5,)), high=3 * to.ones((5,)))
    prior = utils.BoxUniform(**prior_hparam)

    # Posterior (normalizing flow)
    posterior_nn_hparam = dict(model="nsf", embedding_net=nn.Identity(), hidden_features=50, num_transforms=5)

    # Algorithm
    algo_hparam = dict(
        summary_statistic="states",
        max_iter=30,
        num_real_rollouts=num_real_obs,
        num_sim_per_real_rollout=1000,
        num_workers=1,
    )

    sbi_training_hparam = dict(
        training_batch_size=1000,
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
        sbi_training_hparam=sbi_training_hparam,
        **algo_hparam,
    )

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(seed=args.seed),
        dict(prior=prior_hparam),
        dict(posterior_nn=posterior_nn_hparam),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(seed=args.seed)
