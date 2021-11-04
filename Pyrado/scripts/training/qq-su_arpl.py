"""
Train an agent to solve the Qube swing-up task environment using Adversarially Robust Policy Learning.
"""
import torch as to

import pyrado
from pyrado.algorithms.meta.arpl import ARPL
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.environments.pysim.quanser_qube import QQubeSwingUpSim
from pyrado.logger.experiment import save_dicts_to_yaml, setup_experiment
from pyrado.policies.feed_back.fnn import FNNPolicy
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.argparser import get_argparser
from pyrado.utils.data_types import EnvSpec


if __name__ == "__main__":
    # Parse command line arguments
    args = get_argparser().parse_args()

    # Experiment (set seed before creating the modules)
    ex_dir = setup_experiment(QQubeSwingUpSim.name, f"{ARPL.name}_{FNNPolicy.name}", "actnorm")
    # ex_dir = setup_experiment(QQubeSim.name, f'{ARPL.name}_{GRUPolicy.name}', 'actnorm')

    # Set seed if desired
    pyrado.set_seed(args.seed, verbose=True)

    # Environment
    env_hparam = dict(dt=1 / 250.0, max_steps=1500)
    env = QQubeSwingUpSim(**env_hparam)
    env = ActNormWrapper(env)
    env = StateAugmentationWrapper(env, domain_param=None)

    # Policy
    policy_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    env = ARPL.wrap_env(
        env,
        policy,
        dynamics=True,
        process=True,
        observation=True,
        halfspan=0.05,
        dyn_eps=0.07,
        dyn_phi=0.25,
        obs_phi=0.1,
        obs_eps=0.05,
        proc_phi=0.1,
        proc_eps=0.03,
    )

    # Critic
    vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)  # FNN
    # vfcn_hparam = dict(hidden_size=32, num_recurrent_layers=1)  # LSTM & GRU
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    # vfcn = GRUPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.9844534412010116,
        lamda=0.9710614403461155,
        num_epoch=10,
        batch_size=150,
        standardize_adv=False,
        lr=0.00016985313083236645,
    )
    critic = GAE(vfcn, **critic_hparam)

    # Algorithm
    subrtn_hparam = dict(
        max_iter=0,
        min_steps=23 * env.max_steps,
        min_rollouts=None,
        num_workers=12,
        num_epoch=5,
        eps_clip=0.08588362499920563,
        batch_size=150,
        std_init=0.994955464909253,
        lr=0.0001558850276649469,
    )
    algo_hparam = dict(
        max_iter=500,
        steps_num=23 * env.max_steps,
    )
    subrtn = PPO(ex_dir, env, policy, critic, **subrtn_hparam)
    algo = ARPL(ex_dir, env, subrtn, policy, subrtn.expl_strat, **algo_hparam)

    # Save the hyper-parameters
    save_dicts_to_yaml(
        dict(env=env_hparam, seed=args.seed),
        dict(policy=policy_hparam),
        dict(critic=critic_hparam, vfcn=vfcn_hparam),
        dict(subrtn_hparam=subrtn_hparam, subrtn_name=subrtn.name),
        dict(algo=algo_hparam, algo_name=algo.name),
        save_dir=ex_dir,
    )

    # Jeeeha
    algo.train(snapshot_mode="best", seed=args.seed)
