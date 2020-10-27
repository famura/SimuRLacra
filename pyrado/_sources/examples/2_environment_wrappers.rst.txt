How to wrap an environment
--------------------------

Lets first create the basic simulator without any wrappers

.. code-block:: python

    import numpy as np
    from prettyprinter import pprint

    from pyrado.domain_randomization.default_randomizers import get_default_randomizer
    from pyrado.environment_wrappers.action_delay import ActDelayWrapper
    from pyrado.environment_wrappers.action_noise import GaussianActNoiseWrapper
    from pyrado.environment_wrappers.action_normalization import ActNormWrapper
    from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
    from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper
    from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
    from pyrado.environment_wrappers.utils import inner_env, remove_env, typed_env
    from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
    from pyrado.policies.dummy import DummyPolicy
    from pyrado.sampling.rollout import rollout
    from pyrado.utils.data_types import RenderMode

    env = QCartPoleSwingUpSim(dt=1/50., max_steps=10)
    print(f'dim obs: {env.obs_space.flat_dim}\n'
          f'dim state: {env.state_space.flat_dim}\n'
          f'dim act: {env.act_space.flat_dim}\n')

The print reveals that we have a 4-dimensional sate space, a 5-dimensional observation space, and a 1-dimensional
action space, all of type `BoxSpace` which is a straightforward continuous space in $R^n$

Since we are probably here to do some domain randomization, we will start with that.
There are different types of randomizers. The `DomainRandWrapperLive` sets a new set of domain parameters on every
reset of the environment. The `DomainRandWrapperBuffer` maintains a buffer of domain parameter set which we have to
fill explicitly. This way, it cycles through a set of domains. There is also the `MetaDomainRandWrapper` which
adapts the randomizer, i.e. changes the distribution according to which the domain parameters are randomized.

.. code-block:: python

    randomizer = get_default_randomizer(env)
    print(randomizer)
    env_r = DomainRandWrapperBuffer(env, randomizer)
    env_r.fill_buffer(num_domains=3)

Lets have a look at the randomized simulation. Due to the synchronized random seed we have the same initial state as
well as the same random action. However, the trajectory id not the same since the domains evolve differently.
Note that the very first action and reward are only different for logging.

.. code-block:: python

    for i in range(4):
        rollout(env_r, DummyPolicy(env_r.spec), eval=True, seed=0, render_mode=RenderMode(video=False, text=True))
        pprint(env.domain_param, indent=4)

In general, the environments' individual observation dimensions have very different scales and limits.
It is an open secret that one of the most important things for RL to work well are equally scaled actions and
observations. Thus we will scale these spaces to [-1, 1] for every dimension. Check the wrappers' `_process_act` and
`_process_obs` functions for details. One observation dimension does not have a finite lime. Therefore we need to
provide and explicit limit for normalizing. We can also provide explicit bounds to override existing ones.

.. code-block:: python

    print(env_r.act_space)
    env_rn = ActNormWrapper(env)
    print(env_rn.act_space)

    print(env_rn.obs_space)
    elb = {'x_dot': -213., 'theta_dot': -42.}
    eub = {'x_dot': 213., 'theta_dot': 42., 'x': 0.123}
    env_rn = ObsNormWrapper(env_rn, explicit_lb=elb, explicit_ub=eub)
    print(env_rn.obs_space)

So if we now do a rollout, we can see the effect of the normalization

.. code-block:: python

    ro_r = rollout(env_r, DummyPolicy(env_r.spec), eval=True, seed=0, render_mode=RenderMode())
    ro_rn = rollout(env_rn, DummyPolicy(env_rn.spec), eval=True, seed=0, render_mode=RenderMode())
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    print(f'observations without normalization:\n{ro_r.observations}')
    print(f'observations with normalization:\n{ro_rn.observations}')
    assert np.allclose(env_rn._process_obs(ro_r.observations), ro_rn.observations)

In case we want to mask some observations from wer policy, e.g. if the real system does not observe a quantity that is
available during simulation, we can mask them out using the `ObsPartialWrapper`. This wrapper can mask using an array
of zeros and ones, or by passing a list of the exact labels.

.. code-block:: python

    env_rnp = ObsPartialWrapper(env_rn, idcs=['x_dot', 'cos_theta'])
    print(env_rnp.obs_space)
    ro_rnp = rollout(env_rnp, DummyPolicy(env_rnp.spec), eval=True, seed=0, render_mode=RenderMode())
    print(f'partial observations with normalization:\n{ro_rnp.observations}')

We can also apply wrappers that apply additional noise to the action (`GaussianActNoiseWrapper`) or observations
(`GaussianObsNoiseWrapper). The action wrappers will not modify the `action` filed in the recorded rollout, since this
one is capturing the action as they are commanded by the policy.

.. code-block:: python

    env_rnpa = GaussianActNoiseWrapper(env_rnp,
                                       noise_mean=0.5*np.ones(env_rnp.act_space.shape),
                                       noise_std=0.1*np.ones(env_rnp.act_space.shape))
    ro_rnpa = rollout(env_rnpa, DummyPolicy(env_rnpa.spec), eval=True, seed=0, render_mode=RenderMode())
    assert np.allclose(ro_rnp.actions, ro_rnpa.actions)
    assert not np.allclose(ro_rnp.observations, ro_rnpa.observations)

Real-world devices often have delays. One way to model this effect is by artificially hold back the current action for
a given number of time steps. Again, the modified `action` fields in the recorded rollouts are the same. Have a look at
the printed actions `a_t` as well as next state `s_t+1`

.. code-block:: python

    ro_rnp = rollout(env_rnp, DummyPolicy(env_rnp.spec), eval=True, seed=0, render_mode=RenderMode(text=True))  # redo for visual comparison
    env_rnpd = ActDelayWrapper(env_rnp, delay=3)
    ro_rnpd = rollout(env_rnpd, DummyPolicy(env_rnpd.spec), eval=True, seed=0, render_mode=RenderMode(text=True))
    assert np.allclose(ro_rnp.actions, ro_rnpd.actions)
    assert not np.allclose(ro_rnp.observations, ro_rnpd.observations)

There are also very handy utils to manage chains of wrappers. Examples are`inner_env()` which yields the core
environment, `typed_env()` which yields the first element of the chain equal to the provided type, or `remove_env` which
removes the first element of the chain equal to the provided type.

.. code-block:: python

    assert isinstance(inner_env(env_rnpd), QCartPoleSwingUpSim)
    assert typed_env(env_rnpd, ObsPartialWrapper) is not None
    assert isinstance(env_rnpd, ActDelayWrapper)
    env_rnpdr = remove_env(env_rnpd, ActDelayWrapper)
    assert not isinstance(env_rnpdr, ActDelayWrapper)

Finally, **the most important lesson**: the order in which we apply the environment wrappers matters!
For example, applying the `ObsNormWrapper` after the `ObsPartialWrapper` will not give we the intended result (due to
the implementation). Another example is the order of `ObsNormWrapper` and `GaussianObsNoiseWrapper`.
