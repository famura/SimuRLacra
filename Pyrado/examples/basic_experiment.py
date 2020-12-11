"""
This file provides a step-by-step example of how to write a training script in Pyrado.
There are many valid possibilities to deviate from this scheme. However, the following sequence is battle-tested.
"""
import pyrado
from pyrado.algorithms.episodic.hc import HCNormal
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.pysim.ball_on_beam import BallOnBeamSim
from pyrado.logger.experiment import setup_experiment, save_list_of_dicts_to_yaml
from pyrado.policies.features import FeatureStack, identity_feat, sin_feat
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.sampling.rollout import rollout, after_rollout_query
from pyrado.utils.data_types import RenderMode
from pyrado.utils.input_output import print_cbt


"""
Start by creating a new experiment which's folder will be placed in `Pyrado/data/temp` by default. You can change this
by passing any directory as `base_dir`. In Pyrado, the folders are structured like this:
`environment_name/algorithm_name/datetime_and_info`. This rule is only required for the automatic search for experiments
(e.g. used in `sim_policy()`). This search function requires the individual experiment folders to start with `date_time`.
Aside from this, you can name your experiments and folders however you like. Use the `load_experiment()` function to
later oad your results. It will look for an environment as well as a policy file in the provided path.
"""
ex_dir = setup_experiment(BallOnBeamSim.name, f"{HCNormal.name}_{LinearPolicy.name}", "ident-sin")

"""
Additionally, you can set a seed for the random number generators. It is suggested to do so, if you want to
compare changes of certain hyper-parameters to eliminate the effect of the initial state and the initial policy
parameters (both are sampled randomly in most cases).
"""
pyrado.set_seed(seed=0, verbose=True)

"""
Set up the environment a.k.a. domain to train in. After creating the environment, you can apply various wrappers which
are modular. Note that the order of wrappers might be of importance. For example, wrapping an environment with an
`ObsNormWrapper` and then with an `GaussianObsNoiseWrapper` applies the noise on the normalized observations, and yields
different results than the reverse order of wrapping.
Environments in Pyrado can be of different types: (i) written in Python only (like the Quanser simulations or simple
OpenAI Gym environments), (ii) wrapped as well as self-designed MuJoCo-based simulations, or (iii) self-designed
robotic environments powered by Rcs using either the Bullet or Vortex physics engine. None of the simulations includes
any computer vision aspects. It is all about dynamics-based interaction and (continuous) control. The degree of
randomization for the environments varies strongly, since it is a lot of work to randomize them properly (including
testing) and I have to graduate after all ;)
"""
env_hparams = dict(dt=1 / 50.0, max_steps=300)
env = BallOnBeamSim(**env_hparams)
env = ActNormWrapper(env)

"""
Set up the policy after the environment since it needs to know the dimensions of the policies observation and action
space. There are many different policy architectures available under `Pyrado/pyrado/policies`, which significantly
vary in terms of required hyper-parameters. You can find some examples at `Pyrado/scripts/training`.
Note that all policies must inherit from `Policy` which inherits from `torch.nn.Module`. Moreover, all `Policy`
instances are deterministic. The exploration is handled separately (see `Pyrado/pyrado/exploration`).
"""
policy_hparam = dict(feats=FeatureStack([identity_feat, sin_feat]))
policy = LinearPolicy(spec=env.spec, **policy_hparam)

"""
Specify the algorithm you want to use for learning the policy parameters.
For deterministic sampling, you need to set `num_workers=1`. If `num_workers>1`, PyTorch's multiprocessing
library will be used to parallelize sampling from the environment on the CPU. The resulting behavior is non-deterministic,
i.e. even for the same random seed, you will get different results. Moreover, it is advised to set `num_workers` to 1
if you want to debug your code.
The algorithms can be categorized in two different types: one type randomizes the action every step (their exploration
strategy inherits from `StochasticActionExplStrat`), and the other type randomizes the policy parameters once every
rollout their exploration strategy inherits from `StochasticParamExplStrat`). It goes without saying that every
algorithm has different hyper-parameters. However, they all use the same `rollout()` function to generate their data.
"""
algo_hparam = dict(
    max_iter=8,
    pop_size=20,
    num_rollouts=10,
    expl_factor=1.1,
    expl_std_init=1.0,
    num_workers=4,
)
algo = HCNormal(ex_dir, env, policy, **algo_hparam)

"""
Save the hyper-parameters before staring the training in a YAML-file. This step is not strictly necessary, but it helps
you to later see which hyper-parameters you used, i.e. which setting leads to a successfully trained policy.
"""
save_list_of_dicts_to_yaml(
    [dict(env=env_hparams, seed=0), dict(policy=policy_hparam), dict(algo=algo_hparam, algo_name=algo.name)], ex_dir
)

"""
Finally, start the training. The `train()` function is the same for all algorithms inheriting from the `Algorithm`
base class. It repetitively calls the algorithm's custom `step()` and `update()` functions.
You can load and continue a previous experiment using the Algorithm's `load()` method. The `snapshot_mode()` method
determines when to save the current training state, e.g. 'latest' saves after every step of the algorithm, and 'best'
only saves if the average return is a new highscore.
Moreover, you can set the random number generator's seed. This second option for setting the seed comes in handy when
you want to continue from a previous experiment multiple times. 
"""
algo.train(snapshot_mode="latest", seed=None)

input("\nFinished training. Hit enter to simulate the policy.\n")

"""
Simulate the learned policy in the environment it has been trained in. The following is a part of
`scripts/sim_policy.py` which can be executed to simulate any policy given the experiment's directory. 
"""
done, state, param = False, None, None
while not done:
    ro = rollout(
        env,
        policy,
        render_mode=RenderMode(video=True),
        eval=True,
        reset_kwargs=dict(domain_param=param, init_state=state),
    )
    print_cbt(f"Return: {ro.undiscounted_return()}", "g", bright=True)
    done, state, param = after_rollout_query(env, policy, ro)
pyrado.close_vpython()
