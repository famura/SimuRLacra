# Copyright (c) 2020, Fabio Muratore, Honda Research Institute Europe GmbH, and
# Technical University of Darmstadt.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Fabio Muratore, Honda Research Institute Europe GmbH,
#    or Technical University of Darmstadt, nor the names of its contributors may
#    be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL FABIO MURATORE, HONDA RESEARCH INSTITUTE EUROPE GMBH,
# OR TECHNICAL UNIVERSITY OF DARMSTADT BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
# IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import pytest

from pyrado.algorithms.a2c import A2C
from pyrado.algorithms.actor_critic import ActorCritic
from pyrado.algorithms.adr import ADR
from pyrado.algorithms.advantage import GAE
from pyrado.algorithms.arpl import ARPL
from pyrado.algorithms.cem import CEM
from pyrado.algorithms.hc import HCNormal, HCHyper
from pyrado.algorithms.nes import NES
from pyrado.algorithms.parameter_exploring import ParameterExploring
from pyrado.algorithms.pepg import PEPG
from pyrado.algorithms.power import PoWER
from pyrado.algorithms.ppo import PPO, PPO2
from pyrado.algorithms.reps import REPS
from pyrado.algorithms.sac import SAC
from pyrado.algorithms.spota import SPOTA
from pyrado.algorithms.svpg import SVPG
from pyrado.algorithms.sysid_via_episodic_rl import DomainDistrParamPolicy, SysIdViaEpisodicRL
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.domain_randomization.default_randomizers import get_default_randomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer, DomainRandWrapperLive, \
    MetaDomainRandWrapper
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.logger import set_log_prefix_dir
from pyrado.policies.features import *
from pyrado.policies.fnn import FNNPolicy, FNN
from pyrado.policies.rnn import RNNPolicy
from pyrado.policies.linear import LinearPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sequences import *
from pyrado.spaces import ValueFunctionSpace
from pyrado.utils.data_types import EnvSpec
from tests.conftest import m_needs_bullet, policy


# Fixture providing an experiment directory
@pytest.fixture
def ex_dir(tmpdir):
    set_log_prefix_dir(tmpdir)
    return tmpdir


@pytest.mark.longtime
@pytest.mark.parametrize(
    'env', [
        'default_qbb'  # we just need one env to construct the fixture policies
    ],
    ids=['qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'linear_policy',
        'fnn_policy',
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
        'adn_policy',
        'thfnn_policy',
        'thgru_policy',
    ],
    ids=['linear', 'fnn', 'rnn', 'lstm', 'gru', 'adn', 'thfnn', 'thgru'],
    indirect=True
)
@pytest.mark.parametrize(
    'algo_class, algo_hparam', [
        (A2C, dict(std_init=0.1)),
        (PPO, dict(std_init=0.1)),
        (PPO2, dict(std_init=0.1)),
        (HCNormal, dict(expl_std_init=0.1, pop_size=None, expl_factor=1.1)),
        (HCHyper, dict(expl_r_init=0.05, pop_size=None, expl_factor=1.1)),
        (NES, dict(expl_std_init=0.1, pop_size=None)),
        (PEPG, dict(expl_std_init=0.1, pop_size=None)),
        (PoWER, dict(expl_std_init=0.1, pop_size=100, num_is_samples=10)),
        (CEM, dict(expl_std_init=0.1, pop_size=100, num_is_samples=10)),
        # (REPS, dict(eps=0.1, pop_size=500, expl_std_init=0.1)),
    ],
    ids=['a2c', 'ppo', 'ppo2', 'hc_normal', 'hc_hyper', 'nes', 'pepg', 'power', 'cem'])  # , 'reps'])
def test_snapshots_notmeta(ex_dir, env, policy, algo_class, algo_hparam):
    # Collect hyper-parameters, create algorithm, and train
    common_hparam = dict(max_iter=1, num_workers=1)
    common_hparam.update(algo_hparam)

    if issubclass(algo_class, ActorCritic):
        common_hparam.update(min_rollouts=3,
                             critic=GAE(value_fcn=FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace),
                                                            hidden_sizes=[16, 16],
                                                            hidden_nonlin=to.tanh)))
    elif issubclass(algo_class, ParameterExploring):
        common_hparam.update(num_rollouts=1)
    else:
        raise NotImplementedError

    # Train
    algo = algo_class(ex_dir, env, policy, **common_hparam)
    algo.train()
    if isinstance(algo, ActorCritic):
        policy_posttrn_param_values = algo.policy.param_values
        critic_posttrn_value_fcn_param_values = algo.critic.value_fcn.param_values
    elif isinstance(algo, ParameterExploring):
        policy_posttrn_param_values = algo.best_policy_param

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo.load_snapshot(load_dir=ex_dir, meta_info=None)
    policy_loaded = deepcopy(algo.policy)

    # Check
    assert all(policy_posttrn_param_values == policy_loaded.param_values)
    if algo_class in [A2C, PPO, PPO2]:
        critic_loaded = deepcopy(algo.critic)
        assert all(critic_posttrn_value_fcn_param_values == critic_loaded.value_fcn.param_values)


@pytest.mark.parametrize(
    'env', [
        'default_bob'
    ],
    ids=['bob'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'linear_policy'
    ],
    ids=['linear'],
    indirect=True
)
@pytest.mark.parametrize(
    'algo_class, algo_hparam', [
        (HCNormal, dict(expl_std_init=0.1, pop_size=None, expl_factor=1.1)),
        (HCHyper, dict(expl_r_init=0.05, pop_size=None, expl_factor=1.1)),
        (NES, dict(expl_std_init=0.1, pop_size=None)),
        (NES, dict(expl_std_init=0.1, pop_size=None, transform_returns=True)),
        (NES, dict(expl_std_init=0.1, pop_size=None, symm_sampling=True)),
        (PEPG, dict(expl_std_init=0.1, pop_size=None)),
        (PoWER, dict(expl_std_init=0.1, pop_size=100, num_is_samples=10)),
        (CEM, dict(expl_std_init=0.1, pop_size=100, num_is_samples=10, full_cov=True)),
        (CEM, dict(expl_std_init=0.1, pop_size=100, num_is_samples=10, full_cov=False)),
        (REPS, dict(eps=0.1, pop_size=300, expl_std_init=0.1)),
    ],
    ids=['hc_normal', 'hc_hyper', 'nes', 'nes_tr', 'nes_symm', 'pepg', 'power', 'cem-fcov', 'cem-dcov', 'reps']
)
def test_param_expl(ex_dir, env, policy, algo_class, algo_hparam):
    # Hyper-parameters
    common_hparam = dict(max_iter=3, num_rollouts=4)
    common_hparam.update(algo_hparam)

    # Create algorithm and train
    algo = algo_class(ex_dir, env, policy, **common_hparam)
    algo.reset()  # not necessary, but this way we can test it too
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.parametrize(
    'env', [
        'default_bob'
    ],
    ids=['bob'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'linear_policy'
    ],
    ids=['linear'],
    indirect=True
)
@pytest.mark.parametrize(
    'actor_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'value_fcn_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'critic_hparam', [dict(gamma=0.995, lamda=1., num_epoch=1, lr=1e-4, standardize_adv=False)], ids=['casual']
)
@pytest.mark.parametrize(
    'algo_hparam', [dict(max_iter=3, num_particles=3, temperature=10, lr=1e-3, horizon=50)], ids=['casual']
)
def test_svpg(ex_dir, env, policy, actor_hparam, value_fcn_hparam, critic_hparam, algo_hparam):
    # Create algorithm and train
    particle_hparam = dict(actor=actor_hparam, value_fcn=value_fcn_hparam, critic=critic_hparam)
    algo = SVPG(ex_dir, env, particle_hparam, **algo_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.metaalgorithm
@pytest.mark.parametrize(
    'env', [
        'default_qqsu'
    ], ids=['qq-su'],
    indirect=True
)
@pytest.mark.parametrize(
    'subrtn_hparam', [dict(max_iter=3, min_rollouts=5, num_workers=1, num_epoch=4)], ids=['casual']
)
@pytest.mark.parametrize(
    'actor_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'value_fcn_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=['casual']
)
@pytest.mark.parametrize(
    'critic_hparam', [dict(gamma=0.995, lamda=1., num_epoch=1, lr=1e-4, standardize_adv=False)], ids=['casual']
)
@pytest.mark.parametrize(
    'adr_hparam', [dict(max_iter=3, num_svpg_particles=3, num_discriminator_epoch=3, batch_size=100,
                        num_workers=1, randomized_params=[])], ids=['casual']
)
def test_adr(ex_dir, env, subrtn_hparam, actor_hparam, value_fcn_hparam, critic_hparam, adr_hparam):
    # Create the subroutine for the meta-algorithm
    actor = FNNPolicy(spec=env.spec, **actor_hparam)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic = GAE(value_fcn, **critic_hparam)
    subroutine = PPO(ex_dir, env, actor, critic, **subrtn_hparam)

    # Create algorithm and train
    particle_hparam = dict(actor=actor_hparam, value_fcn=value_fcn_hparam, critic=critic_hparam)
    algo = ADR(ex_dir, env, subroutine, svpg_particle_hparam=particle_hparam, **adr_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.longtime
@pytest.mark.metaalgorithm
@pytest.mark.parametrize(
    'env', [
        'default_qbb'
    ], ids=['qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'spota_hparam', [
        dict(max_iter=3, alpha=0.05, beta=0.01, nG=2, nJ=10, ntau=5, nc_init=1, nr_init=1,
             sequence_cand=sequence_add_init, sequence_refs=sequence_const, warmstart_cand=False,
             warmstart_refs=False, num_bs_reps=1000, studentized_ci=False),
    ], ids=['casual_hparam']
)
def test_spota_ppo(ex_dir, env, spota_hparam):
    # Environment and domain randomization
    randomizer = get_default_randomizer(env)
    env = DomainRandWrapperBuffer(env, randomizer)

    # Policy and subroutines
    policy = FNNPolicy(env.spec, [16, 16], hidden_nonlin=to.tanh)
    value_fcn = FNN(input_size=env.obs_space.flat_dim, output_size=1, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    critic_hparam = dict(gamma=0.998, lamda=0.95, num_epoch=3, batch_size=64, lr=1e-3)
    critic_cand = GAE(value_fcn, **critic_hparam)
    critic_refs = GAE(deepcopy(value_fcn), **critic_hparam)

    subrtn_hparam_cand = dict(
        # min_rollouts=0,  # will be overwritten by SPOTA
        min_steps=0,  # will be overwritten by SPOTA
        max_iter=2, num_epoch=3, eps_clip=0.1, batch_size=64, num_workers=1, std_init=0.5, lr=1e-2)
    subrtn_hparam_cand = subrtn_hparam_cand

    sr_cand = PPO(ex_dir, env, policy, critic_cand, **subrtn_hparam_cand)
    sr_refs = PPO(ex_dir, env, deepcopy(policy), critic_refs, **subrtn_hparam_cand)

    # Create algorithm and train
    algo = SPOTA(ex_dir, env, sr_cand, sr_refs, **spota_hparam)
    algo.train()


@pytest.mark.parametrize(
    'env', [
        'default_bob',
        'default_qbb'
    ], ids=['bob', 'qbb'],
    indirect=True
)
@pytest.mark.parametrize(
    'policy', [
        'linear_policy'
    ], ids=['linear'],
    indirect=True
)
@pytest.mark.parametrize(
    'algo, algo_hparam',
    [
        (A2C, dict()),
        (PPO, dict()),
        (PPO2, dict()),
    ],
    ids=['a2c', 'ppo', 'ppo2'])
@pytest.mark.parametrize(
    'value_fcn_type',
    [
        'fnn-plain',
        'fnn',
        'rnn',
    ],
    ids=['vf_fnn_plain', 'vf_fnn', 'vf_rnn']
)
@pytest.mark.parametrize('use_cuda', [False, True], ids=['cpu', 'cuda'])
def test_actor_critic(ex_dir, env, policy, algo, algo_hparam, value_fcn_type, use_cuda):
    # Create value function
    if value_fcn_type == 'fnn-plain':
        value_fcn = FNN(
            input_size=env.obs_space.flat_dim,
            output_size=1,
            hidden_sizes=[16, 16],
            hidden_nonlin=to.tanh,
            use_cuda=use_cuda
        )
    else:
        vf_spec = EnvSpec(env.obs_space, ValueFunctionSpace)
        if value_fcn_type == 'fnn':
            value_fcn = FNNPolicy(
                vf_spec,
                hidden_sizes=[16, 16],
                hidden_nonlin=to.tanh,
                use_cuda=use_cuda
            )
        else:
            value_fcn = RNNPolicy(
                vf_spec,
                hidden_size=16,
                num_recurrent_layers=1,
                use_cuda=use_cuda
            )

    # Create critic
    critic_hparam = dict(
        gamma=0.98,
        lamda=0.95,
        batch_size=32,
        lr=1e-3,
        standardize_adv=False,
    )
    critic = GAE(value_fcn, **critic_hparam)

    # Common hyper-parameters
    common_hparam = dict(max_iter=3, min_rollouts=3, num_workers=1)
    # Add specific hyper parameters if any
    common_hparam.update(algo_hparam)

    # Create algorithm and train
    algo = algo(ex_dir, env, policy, critic, **common_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.longtime
@pytest.mark.parametrize(
    'env', [
        'default_omo'
    ], ids=['omo'],
    indirect=True
)
@pytest.mark.parametrize(
    'algo, algo_hparam', [
        (HCNormal, dict(max_iter=5, pop_size=None, num_rollouts=6, expl_std_init=0.5, expl_factor=1.1)),
        (NES, dict(max_iter=50, pop_size=None, num_rollouts=6, expl_std_init=0.5, symm_sampling=True)),
        (PEPG, dict(max_iter=50, pop_size=100, num_rollouts=6, expl_std_init=0.5, lr=1e-2, normalize_update=False)),
        (PoWER, dict(max_iter=20, pop_size=100, num_rollouts=6, num_is_samples=10, expl_std_init=0.5)),
        (CEM, dict(max_iter=20, pop_size=100, num_rollouts=6, num_is_samples=10, expl_std_init=0.5, full_cov=False)),
        (REPS, dict(max_iter=50, pop_size=500, num_rollouts=6, eps=0.1, expl_std_init=0.5,
                    use_map=True, optim_mode='torch')),
    ], ids=['hc_normal', 'nes', 'pepg', 'power', 'cem', 'reps']
)
def test_training_parameter_exploring(ex_dir, env, algo, algo_hparam):
    # Environment and policy
    env = ActNormWrapper(env)
    policy_hparam = dict(feats=FeatureStack([const_feat, identity_feat]))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Get initial return for comparison
    rets_before = np.zeros(5)
    for i in range(rets_before.size):
        rets_before[i] = rollout(env, policy, eval=True, seed=i).undiscounted_return()

    # Create the algorithm and train
    algo = algo(ex_dir, env, policy, **algo_hparam)
    algo.train()

    # Compare returns before and after training for max_iter iteration
    rets_after = np.zeros_like(rets_before)
    for i in range(rets_before.size):
        rets_after[i] = rollout(env, policy, eval=True, seed=i).undiscounted_return()

    assert all(rets_after > rets_before)


@pytest.mark.parametrize(
    'env', [
        'default_omo'
    ], ids=['omo'],
    indirect=True
)
@pytest.mark.parametrize(
    'module', [
        'linear_policy',
        'fnn_policy',
        'rnn_policy',
        'lstm_policy',
        'gru_policy',
    ]
    , ids=['linear', 'fnn', 'rnn', 'lstm', 'gru'],
    indirect=True
)
def test_soft_update(env, module):
    # Init param values
    target, source = deepcopy(module), deepcopy(module)
    target.param_values = to.zeros_like(target.param_values)
    source.param_values = to.ones_like(source.param_values)

    # Do one soft update
    SAC.soft_update(target, source, tau=0.8)
    assert to.allclose(target.param_values, 0.2*to.ones_like(target.param_values))

    # Do a second soft update to see the exponential decay
    SAC.soft_update(target, source, tau=0.8)
    assert to.allclose(target.param_values, 0.36*to.ones_like(target.param_values))


@pytest.mark.parametrize(
    'env', [
        'default_omo'
    ], ids=['omo'],
    indirect=True
)
def test_arpl(ex_dir, env):
    env = ActNormWrapper(env)
    env = StateAugmentationWrapper(env, domain_param=None)

    policy = FNNPolicy(env.spec, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)

    value_fcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    value_fcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **value_fcn_hparam)
    critic_hparam = dict(
        gamma=0.9844534412010116,
        lamda=0.9710614403461155,
        num_epoch=10,
        batch_size=150,
        standardize_adv=False,
        lr=0.00016985313083236645,
    )
    critic = GAE(value_fcn, **critic_hparam)

    algo_hparam = dict(
        max_iter=0,
        min_steps=23*env.max_steps,
        min_rollouts=None,
        num_workers=1,
        num_epoch=5,
        eps_clip=0.085,
        batch_size=150,
        std_init=0.995,
        lr=2e-4,
    )
    arpl_hparam = dict(
        max_iter=5,
        steps_num=23*env.max_steps,
        halfspan=0.05,
        dyn_eps=0.07,
        dyn_phi=0.25,
        obs_phi=0.1,
        obs_eps=0.05,
        proc_phi=0.1,
        proc_eps=0.03,
        torch_observation=True
    )
    ppo = PPO(ex_dir, env, policy, critic, **algo_hparam)
    algo = ARPL(ex_dir, env, ppo, policy, ppo.expl_strat, **arpl_hparam)

    algo.train(snapshot_mode='best')


@pytest.mark.longtime
@pytest.mark.parametrize(
    'env, num_eval_rollouts', [
        ('default_bob', 5)
    ], ids=['bob'],
    indirect=['env']
)
def test_sysidasrl(ex_dir, env, num_eval_rollouts):
    def eval_ddp_policy(rollouts_real):
        init_states_real = np.array([ro.rollout_info['init_state'] for ro in rollouts_real])
        rollouts_sim = []
        for i, _ in enumerate(range(num_eval_rollouts)):
            rollouts_sim.append(rollout(env_sim, behavior_policy, eval=True,
                                        reset_kwargs=dict(init_state=init_states_real[i, :])))

        # Clip the rollouts rollouts yielding two lists of pairwise equally long rollouts
        ros_real_tr, ros_sim_tr = algo.truncate_rollouts(
            rollouts_real, rollouts_sim, replicate=False
        )
        assert len(ros_real_tr) == len(ros_sim_tr)
        assert all([np.allclose(r.rollout_info['init_state'], s.rollout_info['init_state'])
                    for r, s in zip(ros_real_tr, ros_sim_tr)])

        # Return the average the loss
        losses = [algo.loss_fcn(ro_r, ro_s) for ro_r, ro_s in zip(ros_real_tr, ros_sim_tr)]
        return float(np.mean(np.asarray(losses)))

    # Environments
    env_real = deepcopy(env)
    env_real.domain_param = dict(ang_offset=-2*np.pi/180)

    env_sim = deepcopy(env)
    randomizer = DomainRandomizer(
        UniformDomainParam(name='ang_offset', mean=0, halfspan=1e-12),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {0: ('ang_offset', 'mean'), 1: ('ang_offset', 'halfspan')}
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    assert env_real is not env_sim

    # Policies (the behavioral policy needs to be deterministic)
    behavior_policy = LinearPolicy(env_sim.spec, feats=FeatureStack([identity_feat]))
    prior = DomainRandomizer(
        UniformDomainParam(name='ang_offset', mean=1*np.pi/180, halfspan=1*np.pi/180),
    )
    ddp_policy = DomainDistrParamPolicy(mapping=dp_map, trafo_mask=[False, True], prior=prior)

    # Subroutine
    subrtn_hparam = dict(
        max_iter=5,
        pop_size=40,
        num_rollouts=1,
        num_is_samples=4,
        expl_std_init=1*np.pi/180,
        expl_std_min=0.001,
        extra_expl_std_init=0.,
        extra_expl_decay_iter=5,
        num_workers=1,
    )
    subrtn = CEM(ex_dir, env_sim, ddp_policy, **subrtn_hparam)

    algo_hparam = dict(
        metric=None,
        obs_dim_weight=np.ones(env_sim.obs_space.shape),
        num_rollouts_per_distr=20,
        num_workers=subrtn_hparam['num_workers']
    )

    algo = SysIdViaEpisodicRL(subrtn, behavior_policy, **algo_hparam)

    rollouts_real_tst = []
    for _ in range(num_eval_rollouts):
        rollouts_real_tst.append(rollout(env_real, behavior_policy, eval=True))
    loss_pre = eval_ddp_policy(rollouts_real_tst)

    # Mimic training
    while algo.curr_iter < algo.max_iter and not algo.stopping_criterion_met():
        algo.logger.add_value(algo.iteration_key, algo.curr_iter)

        # Creat fake real-world data
        rollouts_real = []
        for _ in range(num_eval_rollouts):
            rollouts_real.append(rollout(env_real, behavior_policy, eval=True))

        algo.step(snapshot_mode='latest', meta_info=dict(rollouts_real=rollouts_real))

        algo.logger.record_step()
        algo._curr_iter += 1

    loss_post = eval_ddp_policy(rollouts_real_tst)
    assert loss_post <= loss_pre  # don't have to be better every step
