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

from copy import deepcopy

import pytest
from tests.conftest import m_needs_cuda
from tests.environment_wrappers.mock_env import MockEnv

from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.episodic.hc import HCHyper, HCNormal
from pyrado.algorithms.episodic.nes import NES
from pyrado.algorithms.episodic.parameter_exploring import ParameterExploring
from pyrado.algorithms.episodic.pepg import PEPG
from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.episodic.reps import REPS
from pyrado.algorithms.regression.nonlin_regression import NonlinRegression
from pyrado.algorithms.regression.timeseries_prediction import TSPred
from pyrado.algorithms.step_based.a2c import A2C
from pyrado.algorithms.step_based.actor_critic import ActorCritic
from pyrado.algorithms.step_based.dql import DQL
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO, PPO2
from pyrado.algorithms.step_based.sac import SAC
from pyrado.algorithms.step_based.svpg import SVPG
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environments.base import Env
from pyrado.environments.pysim.ball_on_beam import BallOnBeamDiscSim
from pyrado.environments.sim_base import SimEnv
from pyrado.logger import set_log_prefix_dir
from pyrado.policies.base import Policy
from pyrado.policies.features import *
from pyrado.policies.feed_back.fnn import FNN, DiscreteActQValPolicy, FNNPolicy
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.policies.recurrent.rnn import RNNPolicy
from pyrado.policies.recurrent.two_headed_rnn import TwoHeadedGRUPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sequences import *
from pyrado.spaces import BoxSpace, ValueFunctionSpace
from pyrado.spaces.box import InfBoxSpace
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.experiments import load_experiment
from pyrado.utils.functions import noisy_nonlin_fcn


@pytest.fixture
def ex_dir(tmpdir):
    # Fixture providing an experiment directory
    set_log_prefix_dir(tmpdir)
    return tmpdir


@pytest.mark.parametrize(
    "env", ["default_qbb"], ids=["qbb"], indirect=True  # we just need one env to construct the fixture policies
)
@pytest.mark.parametrize(
    "policy",
    [
        "linear_policy",
        "fnn_policy",
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
    ],
    ids=["lin", "fnn", "rnn", "lstm", "gru", "adn"],
    indirect=True,
)
@pytest.mark.parametrize(
    "algo_class, algo_hparam",
    [
        (A2C, dict(std_init=0.1)),
        (PPO, dict(std_init=0.1)),
        (PPO2, dict(std_init=0.1)),
        (HCNormal, dict(expl_std_init=0.1, pop_size=None, expl_factor=1.1)),
        (HCHyper, dict(expl_r_init=0.05, pop_size=None, expl_factor=1.1)),
        (NES, dict(expl_std_init=0.1, pop_size=None)),
        (PEPG, dict(expl_std_init=0.1, pop_size=None)),
        (PoWER, dict(expl_std_init=0.1, pop_size=100, num_is_samples=10)),
        (CEM, dict(expl_std_init=0.1, pop_size=100, num_is_samples=10)),
        (REPS, dict(eps=0.1, pop_size=500, expl_std_init=0.1)),
        (DQL, dict(eps_init=0.2, eps_schedule_gamma=0.99)),
        (SAC, dict()),
    ],
    ids=["a2c", "ppo", "ppo2", "hc_normal", "hc_hyper", "nes", "pepg", "power", "cem", "reps", "dql", "sac"],
)
def test_snapshots_notmeta(ex_dir, env: SimEnv, policy, algo_class, algo_hparam):
    # Collect hyper-parameters, create algorithm, and train
    common_hparam = dict(max_iter=1, num_workers=1)
    common_hparam.update(algo_hparam)

    if issubclass(algo_class, ActorCritic):
        common_hparam.update(
            min_rollouts=3,
            critic=GAE(
                vfcn=FNNPolicy(
                    spec=EnvSpec(env.obs_space, ValueFunctionSpace), hidden_sizes=[16, 16], hidden_nonlin=to.tanh
                )
            ),
        )
    elif issubclass(algo_class, ParameterExploring):
        common_hparam.update(num_init_states_per_domain=1)
    elif issubclass(algo_class, (DQL, SAC)):
        common_hparam.update(memory_size=1000, num_updates_per_step=2, gamma=0.99, min_rollouts=1)
        fnn_hparam = dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)
        if issubclass(algo_class, DQL):
            # Override the setting
            env = BallOnBeamDiscSim(env.dt, env.max_steps)
            net = FNN(
                input_size=DiscreteActQValPolicy.get_qfcn_input_size(env.spec),
                output_size=DiscreteActQValPolicy.get_qfcn_output_size(),
                **fnn_hparam,
            )
            policy = DiscreteActQValPolicy(spec=env.spec, net=net)
        else:
            # Override the setting
            env = ActNormWrapper(env)
            policy = TwoHeadedGRUPolicy(env.spec, shared_hidden_size=8, shared_num_recurrent_layers=1)
            obsact_space = BoxSpace.cat([env.obs_space, env.act_space])
            common_hparam.update(qfcn_1=FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **fnn_hparam))
            common_hparam.update(qfcn_2=FNNPolicy(spec=EnvSpec(obsact_space, ValueFunctionSpace), **fnn_hparam))
    else:
        raise NotImplementedError

    # Simulate training
    algo = algo_class(ex_dir, env, policy, **common_hparam)
    algo.policy.param_values += to.tensor([42.0])
    if isinstance(algo, ActorCritic):
        algo.critic.vfcn.param_values += to.tensor([42.0])

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = Algorithm.load_snapshot(load_dir=ex_dir)
    assert isinstance(algo_loaded, Algorithm)
    policy_loaded = algo_loaded.policy
    if isinstance(algo, ActorCritic):
        critic_loaded = algo_loaded.critic

    # Check
    assert all(algo.policy.param_values == policy_loaded.param_values)
    if isinstance(algo, ActorCritic):
        assert all(algo.critic.vfcn.param_values == critic_loaded.vfcn.param_values)

    # Load the experiment. Since we did not save any hyper-parameters, we ignore the errors when loading.
    env, policy, extra = load_experiment(ex_dir)
    assert isinstance(env, Env)
    assert isinstance(policy, Policy)
    assert isinstance(extra, dict)


@pytest.mark.parametrize("env", ["default_bob"], ids=["bob"], indirect=True)
@pytest.mark.parametrize("policy", ["linear_policy"], ids=["lin"], indirect=True)
@pytest.mark.parametrize(
    "algo_class, algo_hparam",
    [
        (HCNormal, dict(expl_std_init=0.1, pop_size=10, expl_factor=1.1)),
        (HCHyper, dict(expl_r_init=0.05, pop_size=10, expl_factor=1.1)),
        (NES, dict(expl_std_init=0.1, pop_size=10)),
        (NES, dict(expl_std_init=0.1, pop_size=10, transform_returns=True)),
        (NES, dict(expl_std_init=0.1, pop_size=10, symm_sampling=True)),
        (PEPG, dict(expl_std_init=0.1, pop_size=50)),
        (PoWER, dict(expl_std_init=0.1, pop_size=50, num_is_samples=10)),
        (CEM, dict(expl_std_init=0.1, pop_size=50, num_is_samples=10, full_cov=True)),
        (CEM, dict(expl_std_init=0.1, pop_size=50, num_is_samples=10, full_cov=False)),
        (REPS, dict(eps=1.0, pop_size=50, expl_std_init=0.1)),
    ],
    ids=["hc_normal", "hc_hyper", "nes", "nes_tr", "nes_symm", "pepg", "power", "cem-fcov", "cem-dcov", "reps"],
)
def test_param_expl(ex_dir, env, policy, algo_class, algo_hparam):
    pyrado.set_seed(0)

    # Hyper-parameters
    common_hparam = dict(max_iter=2, num_init_states_per_domain=4, num_workers=1)
    common_hparam.update(algo_hparam)

    # Create algorithm and train
    algo = algo_class(ex_dir, env, policy, **common_hparam)
    algo.reset()  # not necessary, but this way we can test it too
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.parametrize("env", ["default_bob"], ids=["bob"], indirect=True)
@pytest.mark.parametrize("policy", ["linear_policy"], ids=["lin"], indirect=True)
@pytest.mark.parametrize("actor_hparam", [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=["casual"])
@pytest.mark.parametrize("vfcn_hparam", [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)], ids=["casual"])
@pytest.mark.parametrize(
    "critic_hparam", [dict(gamma=0.995, lamda=1.0, num_epoch=1, lr=1e-4, standardize_adv=False)], ids=["casual"]
)
@pytest.mark.parametrize(
    "algo_hparam",
    [dict(max_iter=2, num_particles=3, temperature=10, lr=1e-3, horizon=50, num_workers=1)],
    ids=["casual"],
)
def test_svpg(ex_dir, env: SimEnv, policy, actor_hparam, vfcn_hparam, critic_hparam, algo_hparam):
    # Create algorithm and train
    particle_hparam = dict(actor=actor_hparam, vfcn=vfcn_hparam, critic=critic_hparam)
    algo = SVPG(ex_dir, env, particle_hparam, **algo_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.parametrize("env", ["default_bob", "default_qbb"], ids=["bob", "qbb"], indirect=True)
@pytest.mark.parametrize("policy", ["dummy_policy"], indirect=True)
@pytest.mark.parametrize(
    "algo, algo_hparam",
    [(A2C, dict()), (PPO, dict()), (PPO2, dict())],
    ids=["a2c", "ppo", "ppo2"],
)
@pytest.mark.parametrize(
    "vfcn_type",
    ["fnn-plain", FNNPolicy.name, RNNPolicy.name],
    ids=["vf_fnn_plain", "vf_fnn", "vf_rnn"],
)
@pytest.mark.parametrize("use_cuda", [False, pytest.param(True, marks=m_needs_cuda)], ids=["cpu", "cuda"])
def test_actor_critic(ex_dir, env: SimEnv, policy: Policy, algo, algo_hparam, vfcn_type, use_cuda):
    pyrado.set_seed(0)

    if use_cuda:
        policy._device = "cuda"
        policy = policy.to(device="cuda")

    # Create value function
    if vfcn_type == "fnn-plain":
        vfcn = FNN(
            input_size=env.obs_space.flat_dim,
            output_size=1,
            hidden_sizes=[16, 16],
            hidden_nonlin=to.tanh,
            use_cuda=use_cuda,
        )
    elif vfcn_type == FNNPolicy.name:
        vf_spec = EnvSpec(env.obs_space, ValueFunctionSpace)
        vfcn = FNNPolicy(vf_spec, hidden_sizes=[16, 16], hidden_nonlin=to.tanh, use_cuda=use_cuda)
    elif vfcn_type == RNNPolicy.name:
        vf_spec = EnvSpec(env.obs_space, ValueFunctionSpace)
        vfcn = RNNPolicy(vf_spec, hidden_size=16, num_recurrent_layers=1, use_cuda=use_cuda)
    else:
        raise NotImplementedError

    # Create critic
    critic_hparam = dict(
        gamma=0.98,
        lamda=0.95,
        batch_size=32,
        lr=1e-3,
        standardize_adv=False,
    )
    critic = GAE(vfcn, **critic_hparam)

    # Common hyper-parameters
    common_hparam = dict(max_iter=2, min_rollouts=3, num_workers=1)
    # Add specific hyper parameters if any
    common_hparam.update(algo_hparam)

    # Create algorithm and train
    algo = algo(ex_dir, env, policy, critic, **common_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_bob"], ids=["bob"], indirect=True)
@pytest.mark.parametrize(
    "algo, algo_hparam",
    [
        (
            HCNormal,
            dict(
                max_iter=5,
                pop_size=50,
                num_init_states_per_domain=4,
                expl_std_init=0.5,
                expl_factor=1.1,
            ),
        ),
        (
            PEPG,
            dict(
                max_iter=40,
                pop_size=200,
                num_init_states_per_domain=10,
                expl_std_init=0.2,
                lr=1e-2,
                normalize_update=False,
            ),
        ),
        (
            NES,
            dict(
                max_iter=5,
                pop_size=50,
                num_init_states_per_domain=4,
                expl_std_init=0.5,
                symm_sampling=True,
                eta_mean=2,
            ),
        ),
        (
            PoWER,
            dict(
                max_iter=5,
                pop_size=50,
                num_init_states_per_domain=4,
                num_is_samples=8,
                expl_std_init=0.5,
            ),
        ),
        (
            CEM,
            dict(
                max_iter=5,
                pop_size=50,
                num_init_states_per_domain=4,
                num_is_samples=8,
                expl_std_init=0.5,
                full_cov=False,
            ),
        ),
        (
            REPS,
            dict(
                max_iter=5,
                pop_size=50,
                num_init_states_per_domain=4,
                eps=1.5,
                expl_std_init=0.5,
                use_map=True,
            ),
        ),
    ],
    ids=["hc_normal", "pepg", "nes", "power", "cem", "reps"],
)
def test_training_parameter_exploring(ex_dir, env: SimEnv, algo, algo_hparam):
    # Environment and policy
    env = ActNormWrapper(env)
    policy_hparam = dict(feats=FeatureStack(const_feat, identity_feat))
    policy = LinearPolicy(spec=env.spec, **policy_hparam)

    # Get initial return for comparison
    rets_before = np.zeros(5)
    for i in range(rets_before.size):
        rets_before[i] = rollout(env, policy, eval=True, seed=i).undiscounted_return()

    # Create the algorithm and train
    algo_hparam["num_workers"] = 1
    algo = algo(ex_dir, env, policy, **algo_hparam)
    algo.train()
    policy.param_values = algo.best_policy_param  # mimic saving and loading

    # Compare returns before and after training for max_iter iteration
    rets_after = np.zeros_like(rets_before)
    for i in range(rets_before.size):
        rets_after[i] = rollout(env, policy, eval=True, seed=i).undiscounted_return()

    assert all(rets_after > rets_before)


@pytest.mark.parametrize("env", ["default_omo"], ids=["omo"], indirect=True)
@pytest.mark.parametrize(
    "policy",
    [
        "linear_policy",
        "fnn_policy",
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
    ],
    ids=["lin", "fnn", "rnn", "lstm", "gru"],
    indirect=True,
)
def test_soft_update(env, policy: Policy):
    # Init param values
    target, source = deepcopy(policy), deepcopy(policy)
    target.param_values = to.zeros_like(target.param_values)
    source.param_values = to.ones_like(source.param_values)

    # Do one soft update
    SAC.soft_update(target, source, tau=0.8)
    assert to.allclose(target.param_values, 0.2 * to.ones_like(target.param_values))

    # Do a second soft update to see the exponential decay
    SAC.soft_update(target, source, tau=0.8)
    assert to.allclose(target.param_values, 0.36 * to.ones_like(target.param_values))


@pytest.mark.visual
@pytest.mark.parametrize("num_feat_per_dim", [1000], ids=[1000])
@pytest.mark.parametrize("loss_fcn", [to.nn.MSELoss()], ids=["mse"])
@pytest.mark.parametrize("algo_hparam", [dict(max_iter=50, max_iter_no_improvement=5)], ids=["casual"])
def test_rff_regression(ex_dir, num_feat_per_dim: int, loss_fcn: Callable, algo_hparam: dict):
    # Generate some data
    inputs = to.linspace(-4.0, 4.0, 8001).view(-1, 1)
    targets = noisy_nonlin_fcn(inputs, f=3.0, noise_std=0).view(-1, 1)

    # Create the policy
    rff = RFFeat(inp_dim=1, num_feat_per_dim=num_feat_per_dim, bandwidth=1 / 20)
    policy = LinearPolicy(EnvSpec(InfBoxSpace(shape=(1,)), InfBoxSpace(shape=(1,))), FeatureStack(rff))

    # Create the algorithm, and train
    loss_before = loss_fcn(policy(inputs), targets)
    algo = NonlinRegression(ex_dir, inputs, targets, policy, **algo_hparam)
    algo.train()
    loss_after = loss_fcn(policy(inputs), targets)
    assert loss_after < loss_before
    assert algo.curr_iter >= algo_hparam["max_iter_no_improvement"]


@pytest.mark.recurrent_policy
@pytest.mark.parametrize("env", [MockEnv(obs_space=InfBoxSpace(shape=1), act_space=InfBoxSpace(shape=1))])
@pytest.mark.parametrize(
    "policy",
    [
        # Two-headed policies are not supported
        "rnn_policy",
        "lstm_policy",
        "gru_policy",
        "adn_policy",
        "nf_policy",
    ],
    ids=["rnn", "lstm", "gru", "adn", "nf"],
    indirect=True,
)
@pytest.mark.parametrize("windowed", [False, True], ids=["windowed", "not_windowed"])
@pytest.mark.parametrize("cascaded", [False, True], ids=["cascaded", "not_cascaded"])
def test_time_series_prediction(ex_dir, dataset_ts, env: MockEnv, policy: Policy, windowed: bool, cascaded: bool):
    algo_hparam = dict(
        max_iter=1, windowed=windowed, cascaded=cascaded, optim_hparam=dict(lr=1e-2, eps=1e-8, weight_decay=1e-4)
    )
    algo = TSPred(ex_dir, dataset_ts, policy, **algo_hparam)

    # Train
    algo.train()
    assert algo.curr_iter == 1

    if windowed:
        inp_seq = dataset_ts.data_trn_inp
        targ_seq = dataset_ts.data_trn_targ
    else:
        inp_seq = dataset_ts.data_trn_ws
        targ_seq = dataset_ts.data_trn_ws

    # Make the predictions
    preds, hidden = TSPred.predict(policy, inp_seq, windowed, cascaded, hidden=None)

    # Check types
    assert isinstance(preds, to.Tensor)
    assert isinstance(hidden, to.Tensor)
    # Check sizes
    if windowed:
        assert preds.shape[0] == 1
    else:
        assert preds.shape[0] == inp_seq.shape[0]
    assert preds.shape[1] == env.spec.act_space.flat_dim
    assert hidden.numel() == policy.hidden_size

    preds, loss = TSPred.evaluate(
        policy, inp_seq, targ_seq, windowed, cascaded, num_init_samples=2, hidden=None, verbose=False
    )
    assert isinstance(preds, to.Tensor)
    assert isinstance(loss, to.Tensor)
