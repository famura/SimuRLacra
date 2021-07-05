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

from collections import namedtuple
from copy import deepcopy
from typing import List, Optional

import pytest
import sbi.utils as sbiutils
import torch as to
import torch.nn as nn
from sbi.inference import SNPE_C

import pyrado
from pyrado.algorithms.base import Algorithm
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.episodic.hc import HCNormal
from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.episodic.reps import REPS
from pyrado.algorithms.episodic.sysid_via_episodic_rl import DomainDistrParamPolicy, SysIdViaEpisodicRL
from pyrado.algorithms.meta.arpl import ARPL
from pyrado.algorithms.meta.bayessim import BayesSim
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.algorithms.meta.epopt import EPOpt
from pyrado.algorithms.meta.npdr import NPDR
from pyrado.algorithms.meta.pddr import PDDR
from pyrado.algorithms.meta.simopt import SimOpt
from pyrado.algorithms.meta.spota import SPOTA
from pyrado.algorithms.meta.sprl import SPRL
from pyrado.algorithms.meta.udr import UDR
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.domain_randomization.default_randomizers import (
    create_default_domain_param_map_qq,
    create_default_randomizer,
    create_default_randomizer_qbb,
    create_zero_var_randomizer,
)
from pyrado.domain_randomization.domain_parameter import NormalDomainParam, SelfPacedDomainParam, UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import (
    DomainRandWrapperBuffer,
    DomainRandWrapperLive,
    MetaDomainRandWrapper,
)
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.base import Env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger import set_log_prefix_dir
from pyrado.policies.base import Policy
from pyrado.policies.features import FeatureStack, identity_feat
from pyrado.policies.feed_back.fnn import FNN, FNNPolicy
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sbi_embeddings import (
    AllStepsEmbedding,
    BayesSimEmbedding,
    DeltaStepsEmbedding,
    DynamicTimeWarpingEmbedding,
    LastStepEmbedding,
    RNNEmbedding,
)
from pyrado.sampling.sbi_rollout_sampler import RolloutSamplerForSBI, SimRolloutSamplerForSBI
from pyrado.sampling.sequences import *
from pyrado.spaces import BoxSpace, ValueFunctionSpace
from pyrado.utils.argparser import MockArgs
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.experiments import load_experiment


@pytest.fixture
def ex_dir(tmpdir):
    # Fixture providing an experiment directory
    set_log_prefix_dir(tmpdir)
    return tmpdir


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_qbb"], indirect=True)
@pytest.mark.parametrize(
    "spota_hparam",
    [
        dict(
            max_iter=2,
            alpha=0.05,
            beta=0.01,
            nG=2,
            nJ=2,
            ntau=2,
            nc_init=1,
            nr_init=1,
            sequence_cand=sequence_add_init,
            sequence_refs=sequence_const,
            warmstart_cand=False,
            warmstart_refs=False,
            num_bs_reps=100,
            studentized_ci=False,
        )
    ],
)
def test_spota_hc(ex_dir, env: SimEnv, spota_hparam: dict):
    pyrado.set_seed(0)

    # Environment and domain randomization
    randomizer = create_default_randomizer(env)
    env = DomainRandWrapperBuffer(env, randomizer)

    # Policy and subroutines
    policy = LinearPolicy(env.spec, feats=FeatureStack(identity_feat))

    subrtn_hparam_common = dict(
        max_iter=1,
        num_init_states_per_domain=1,
        num_domains=1,  # will be overwritten by SPOTA
        expl_factor=1.1,
        expl_std_init=0.5,
        num_workers=1,
    )

    sr_cand = HCNormal(ex_dir, env, policy, **subrtn_hparam_common)
    sr_refs = HCNormal(ex_dir, env, deepcopy(policy), **subrtn_hparam_common)

    # Create algorithm and train
    algo = SPOTA(ex_dir, env, sr_cand, sr_refs, **spota_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter or algo.stopping_criterion_met()

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_qbb"], indirect=True)
@pytest.mark.parametrize(
    "spota_hparam",
    [
        dict(
            max_iter=2,
            alpha=0.05,
            beta=0.01,
            nG=2,
            nJ=2,
            ntau=2,
            nc_init=2,
            nr_init=2,
            sequence_cand=sequence_rec_double,
            sequence_refs=sequence_rec_sqrt,
            warmstart_cand=True,
            warmstart_refs=True,
            num_bs_reps=100,
            studentized_ci=True,
        ),
    ],
)
def test_spota_ppo(ex_dir, env: SimEnv, spota_hparam: dict):
    pyrado.set_seed(0)

    # Environment and domain randomization
    randomizer = create_default_randomizer(env)
    env = DomainRandWrapperBuffer(env, randomizer)

    # Policy and subroutines
    policy = FNNPolicy(env.spec, [16, 16], hidden_nonlin=to.tanh)
    vfcn = FNN(input_size=env.obs_space.flat_dim, output_size=1, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    critic_hparam = dict(gamma=0.998, lamda=0.95, num_epoch=3, batch_size=64, lr=1e-3)
    critic_cand = GAE(vfcn, **critic_hparam)
    critic_refs = GAE(deepcopy(vfcn), **critic_hparam)

    subrtn_hparam_common = dict(
        # min_rollouts=0,  # will be overwritten by SPOTA
        min_steps=0,  # will be overwritten by SPOTA
        max_iter=1,
        num_epoch=3,
        eps_clip=0.1,
        batch_size=64,
        num_workers=1,
        std_init=0.5,
        lr=1e-2,
    )

    sr_cand = PPO(ex_dir, env, policy, critic_cand, **subrtn_hparam_common)
    sr_refs = PPO(ex_dir, env, deepcopy(policy), critic_refs, **subrtn_hparam_common)

    # Create algorithm and train
    algo = SPOTA(ex_dir, env, sr_cand, sr_refs, **spota_hparam)
    algo.train()
    assert algo.curr_iter == algo.max_iter or algo.stopping_criterion_met()

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)
    assert all(algo.policy.param_values == algo.subroutine_cand.policy.param_values)


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_qqsu"], ids=["qqsu"], indirect=True)
@pytest.mark.parametrize(
    "bayrn_hparam",
    [
        dict(
            max_iter=2,
            acq_fc="UCB",
            acq_param=dict(beta=0.25),
            acq_restarts=50,
            acq_samples=50,
            num_init_cand=2,
            warmstart=False,
            mc_estimator=True,
            num_eval_rollouts_sim=3,
            num_eval_rollouts_real=2,  # sim-2-sim
        ),
    ],
)
def test_bayrn_ppo(ex_dir, env: SimEnv, bayrn_hparam: dict):
    pyrado.set_seed(0)

    # Environments and domain randomization
    env_real = deepcopy(env)
    env_sim = DomainRandWrapperLive(env, create_zero_var_randomizer(env))
    dp_map = create_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)
    env_real.domain_param = dict(mass_pend_pole=0.024 * 1.1, mass_rot_pole=0.095 * 1.1)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy and subroutine
    policy = FNNPolicy(env.spec, [16, 16], hidden_nonlin=to.tanh)
    vfcn = FNN(input_size=env.obs_space.flat_dim, output_size=1, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    critic_hparam = dict(gamma=0.998, lamda=0.95, num_epoch=3, batch_size=64, lr=1e-3)
    critic = GAE(vfcn, **critic_hparam)
    subrtn_hparam = dict(
        max_iter=1,
        min_rollouts=3,
        num_epoch=2,
        eps_clip=0.1,
        batch_size=64,
        num_workers=1,
        std_init=0.5,
        lr=1e-2,
    )
    subrtn = PPO(ex_dir, env_sim, policy, critic, **subrtn_hparam)

    # Set the boundaries for the GP
    dp_nom = inner_env(env_sim).get_nominal_domain_param()
    ddp_space = BoxSpace(
        bound_lo=np.array([0.8 * dp_nom["mass_pend_pole"], 1e-8, 0.8 * dp_nom["mass_rot_pole"], 1e-8]),
        bound_up=np.array([1.2 * dp_nom["mass_pend_pole"], 1e-7, 1.2 * dp_nom["mass_rot_pole"], 1e-7]),
    )

    # Create algorithm and train
    algo = BayRn(ex_dir, env_sim, env_real, subrtn, ddp_space, **bayrn_hparam, num_workers=1)
    algo.train()
    assert algo.curr_iter == algo.max_iter or algo.stopping_criterion_met()

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_qqsu"], ids=["qqsu"], indirect=True)
@pytest.mark.parametrize(
    "bayrn_hparam",
    [
        dict(
            max_iter=2,
            acq_fc="UCB",
            acq_param=dict(beta=0.25),
            acq_restarts=50,
            acq_samples=50,
            num_init_cand=2,
            warmstart=False,
            mc_estimator=True,
            num_eval_rollouts_sim=3,
            num_eval_rollouts_real=2,  # sim-2-sim
        ),
        dict(
            max_iter=2,
            acq_fc="EI",
            acq_restarts=50,
            acq_samples=50,
            num_init_cand=2,
            warmstart=True,
            mc_estimator=False,
            thold_succ=40.0,
            thold_succ_subrtn=50.0,
            num_eval_rollouts_sim=3,
            num_eval_rollouts_real=2,  # sim-2-sim
        ),
    ],
    ids=["configUCB", "configEI"],
)
def test_bayrn_power(ex_dir, env: SimEnv, bayrn_hparam: dict):
    pyrado.set_seed(0)

    # Environments and domain randomization
    env_real = deepcopy(env)
    env_sim = DomainRandWrapperLive(env, create_zero_var_randomizer(env))
    dp_map = create_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)
    env_real.domain_param = dict(mass_pend_pole=0.024 * 1.1, mass_rot_pole=0.095 * 1.1)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy and subroutine
    policy_hparam = dict(energy_gain=0.587, ref_energy=0.827)
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec, **policy_hparam)
    subrtn_hparam = dict(
        max_iter=1,
        pop_size=6,
        num_init_states_per_domain=1,
        num_is_samples=4,
        expl_std_init=0.1,
        num_workers=1,
    )
    subrtn = PoWER(ex_dir, env_sim, policy, **subrtn_hparam)

    # Set the boundaries for the GP
    dp_nom = inner_env(env_sim).get_nominal_domain_param()
    ddp_space = BoxSpace(
        bound_lo=np.array([0.8 * dp_nom["mass_pend_pole"], 1e-8, 0.8 * dp_nom["mass_rot_pole"], 1e-8]),
        bound_up=np.array([1.2 * dp_nom["mass_pend_pole"], 1e-7, 1.2 * dp_nom["mass_rot_pole"], 1e-7]),
    )

    # Create algorithm and train
    algo = BayRn(ex_dir, env_sim, env_real, subrtn, ddp_space, **bayrn_hparam, num_workers=1)
    algo.train()
    assert algo.curr_iter == algo.max_iter or algo.stopping_criterion_met()

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)


@pytest.mark.parametrize("env", ["default_omo"], ids=["omo"], indirect=True)
def test_arpl(ex_dir, env: SimEnv):
    pyrado.set_seed(0)

    env = ActNormWrapper(env)
    env = StateAugmentationWrapper(env, domain_param=None)

    policy = FNNPolicy(env.spec, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)

    vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.tanh)
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.9844534412010116,
        lamda=0.9710614403461155,
        num_epoch=10,
        batch_size=150,
        standardize_adv=False,
        lr=0.00016985313083236645,
    )
    critic = GAE(vfcn, **critic_hparam)

    algo_hparam = dict(
        max_iter=2,
        min_steps=23 * env.max_steps,
        min_rollouts=None,
        num_epoch=5,
        eps_clip=0.085,
        batch_size=150,
        std_init=0.995,
        lr=2e-4,
        num_workers=1,
    )
    arpl_hparam = dict(
        max_iter=2,
        steps_num=23 * env.max_steps,
        halfspan=0.05,
        dyn_eps=0.07,
        dyn_phi=0.25,
        obs_phi=0.1,
        obs_eps=0.05,
        proc_phi=0.1,
        proc_eps=0.03,
        torch_observation=True,
    )
    ppo = PPO(ex_dir, env, policy, critic, **algo_hparam)
    algo = ARPL(ex_dir, env, ppo, policy, ppo.expl_strat, **arpl_hparam)

    algo.train(snapshot_mode="best")
    assert algo.curr_iter == algo.max_iter

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)


@pytest.mark.slow
@pytest.mark.parametrize("env, num_eval_rollouts", [("default_bob", 5)], ids=["bob"], indirect=["env"])
def test_sysidasrl_reps(ex_dir, env: SimEnv, num_eval_rollouts: int):
    pyrado.set_seed(0)

    def eval_ddp_policy(rollouts_real):
        init_states_real = np.array([ro.states[0, :] for ro in rollouts_real])
        rollouts_sim = []
        for i, _ in enumerate(range(num_eval_rollouts)):
            rollouts_sim.append(
                rollout(env_sim, behavior_policy, eval=True, reset_kwargs=dict(init_state=init_states_real[i, :]))
            )

        # Clip the rollouts rollouts yielding two lists of pairwise equally long rollouts
        ros_real_tr, ros_sim_tr = algo.truncate_rollouts(rollouts_real, rollouts_sim, replicate=False)
        assert len(ros_real_tr) == len(ros_sim_tr)
        assert all([np.allclose(r.states[0, :], s.states[0, :]) for r, s in zip(ros_real_tr, ros_sim_tr)])

        # Return the average the loss
        losses = [algo.loss_fcn(ro_r, ro_s) for ro_r, ro_s in zip(ros_real_tr, ros_sim_tr)]
        return float(np.mean(np.asarray(losses)))

    # Environments
    env_real = deepcopy(env)
    env_real.domain_param = dict(ang_offset=-2 * np.pi / 180)

    env_sim = deepcopy(env)
    randomizer = DomainRandomizer(
        UniformDomainParam(name="ang_offset", mean=0, halfspan=1e-6),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {0: ("ang_offset", "mean"), 1: ("ang_offset", "halfspan")}
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    assert env_real is not env_sim

    # Policies (the behavioral policy needs to be deterministic)
    behavior_policy = LinearPolicy(env_sim.spec, feats=FeatureStack(identity_feat))
    prior = DomainRandomizer(
        UniformDomainParam(name="ang_offset", mean=1 * np.pi / 180, halfspan=1 * np.pi / 180),
    )
    ddp_policy = DomainDistrParamPolicy(mapping=dp_map, trafo_mask=[False, True], prior=prior)

    # Subroutine
    subrtn_hparam = dict(
        max_iter=2,
        eps=1.0,
        pop_size=100,
        num_init_states_per_domain=1,
        expl_std_init=5e-2,
        expl_std_min=1e-4,
        num_workers=1,
    )
    subrtn = REPS(ex_dir, env_sim, ddp_policy, **subrtn_hparam)

    algo_hparam = dict(
        metric=None, obs_dim_weight=np.ones(env_sim.obs_space.shape), num_rollouts_per_distr=5, num_workers=1
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

        algo.step(snapshot_mode="latest", meta_info=dict(rollouts_real=rollouts_real))

        algo.logger.record_step()
        algo._curr_iter += 1

    loss_post = eval_ddp_policy(rollouts_real_tst)
    assert loss_post <= loss_pre  # don't have to be better every step


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_qqsu"], ids=["qqsu"], indirect=True)
def test_simopt_cem_ppo(ex_dir, env: SimEnv):
    pyrado.set_seed(0)

    # Environments
    env_real = deepcopy(env)
    env_real = ActNormWrapper(env_real)
    env_sim = ActNormWrapper(env)
    randomizer = DomainRandomizer(
        NormalDomainParam(name="mass_rot_pole", mean=0.0, std=1e6, clip_lo=1e-3),
        NormalDomainParam(name="mass_pend_pole", mean=0.0, std=1e6, clip_lo=1e-3),
        NormalDomainParam(name="length_rot_pole", mean=0.0, std=1e6, clip_lo=1e-3),
        NormalDomainParam(name="length_pend_pole", mean=0.0, std=1e6, clip_lo=1e-3),
    )
    env_sim = DomainRandWrapperLive(env_sim, randomizer)
    dp_map = {
        0: ("mass_rot_pole", "mean"),
        1: ("mass_rot_pole", "std"),
        2: ("mass_pend_pole", "mean"),
        3: ("mass_pend_pole", "std"),
        4: ("length_rot_pole", "mean"),
        5: ("length_rot_pole", "std"),
        6: ("length_pend_pole", "mean"),
        7: ("length_pend_pole", "std"),
    }
    trafo_mask = [True] * 8
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)

    # Subroutine for policy improvement
    behav_policy_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    behav_policy = FNNPolicy(spec=env_sim.spec, **behav_policy_hparam)
    vfcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.relu)
    vfcn = FNNPolicy(spec=EnvSpec(env_sim.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.99,
        lamda=0.98,
        num_epoch=2,
        batch_size=128,
        standardize_adv=True,
        lr=8e-4,
        max_grad_norm=5.0,
    )
    critic = GAE(vfcn, **critic_hparam)
    subrtn_policy_hparam = dict(
        max_iter=2,
        eps_clip=0.13,
        min_steps=4 * env_sim.max_steps,
        num_epoch=3,
        batch_size=128,
        std_init=0.75,
        lr=3e-04,
        max_grad_norm=1.0,
        num_workers=1,
    )
    subrtn_policy = PPO(ex_dir, env_sim, behav_policy, critic, **subrtn_policy_hparam)

    prior = DomainRandomizer(
        NormalDomainParam(name="mass_rot_pole", mean=0.095, std=0.095 / 10),
        NormalDomainParam(name="mass_pend_pole", mean=0.024, std=0.024 / 10),
        NormalDomainParam(name="length_rot_pole", mean=0.085, std=0.085 / 10),
        NormalDomainParam(name="length_pend_pole", mean=0.129, std=0.129 / 10),
    )
    ddp_policy_hparam = dict(mapping=dp_map, trafo_mask=trafo_mask, scale_params=True)
    ddp_policy = DomainDistrParamPolicy(prior=prior, **ddp_policy_hparam)
    subsubrtn_distr_hparam = dict(
        max_iter=2,
        pop_size=10,
        num_init_states_per_domain=1,
        num_is_samples=8,
        expl_std_init=1e-2,
        expl_std_min=1e-5,
        extra_expl_std_init=1e-2,
        extra_expl_decay_iter=5,
        num_workers=1,
    )
    subsubrtn_distr = CEM(ex_dir, env_sim, ddp_policy, **subsubrtn_distr_hparam)
    subrtn_distr_hparam = dict(
        metric=None,
        obs_dim_weight=[1, 1, 1, 1, 10, 10],
        num_rollouts_per_distr=3,
        num_workers=1,
    )
    subrtn_distr = SysIdViaEpisodicRL(subsubrtn_distr, behavior_policy=behav_policy, **subrtn_distr_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_eval_rollouts=5,
        warmstart=True,
    )
    algo = SimOpt(ex_dir, env_sim, env_real, subrtn_policy, subrtn_distr, **algo_hparam)

    algo.train()
    assert algo.curr_iter == algo.max_iter

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)


@pytest.mark.parametrize("env", ["default_qbb"], ids=["qbb"], indirect=True)
@pytest.mark.parametrize("policy", ["linear_policy"], ids=["lin"], indirect=True)
@pytest.mark.parametrize(
    "algo, algo_hparam",
    [(UDR, {}), (EPOpt, dict(skip_iter=2, epsilon=0.2, gamma=0.9995))],
    ids=["udr", "EPOpt"],
)
def test_basic_meta(ex_dir, policy, env: SimEnv, algo, algo_hparam: dict):
    pyrado.set_seed(0)

    # Policy and subroutine
    env = GaussianObsNoiseWrapper(
        env,
        noise_std=[
            1 / 180 * np.pi,
            1 / 180 * np.pi,
            0.0025,
            0.0025,
            2 / 180 * np.pi,
            2 / 180 * np.pi,
            0.05,
            0.05,
        ],
    )
    env = ActNormWrapper(env)
    env = ActDelayWrapper(env)
    randomizer = create_default_randomizer_qbb()
    randomizer.add_domain_params(UniformDomainParam(name="act_delay", mean=15, halfspan=15, clip_lo=0, roundint=True))
    env = DomainRandWrapperLive(env, randomizer)

    # Policy
    policy_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)  # FNN
    policy = FNNPolicy(spec=env.spec, **policy_hparam)

    # Critic
    vfcn_hparam = dict(hidden_sizes=[16, 16], hidden_nonlin=to.tanh)  # FNN
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.9995,
        lamda=0.98,
        num_epoch=2,
        batch_size=64,
        lr=5e-4,
        standardize_adv=False,
    )
    critic = GAE(vfcn, **critic_hparam)

    subrtn_hparam = dict(
        max_iter=3,
        min_rollouts=5,
        num_epoch=2,
        eps_clip=0.1,
        batch_size=64,
        std_init=0.8,
        lr=2e-4,
        num_workers=1,
    )
    subrtn = PPO(ex_dir, env, policy, critic, **subrtn_hparam)
    algo = algo(env, subrtn, **algo_hparam)

    algo.train()
    assert algo.curr_iter == algo.max_iter

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)


@pytest.mark.parametrize("env", ["default_qqsu"], ids=["qqsu"], indirect=True)
@pytest.mark.parametrize(
    "embedding_name",
    [
        LastStepEmbedding.name,
        AllStepsEmbedding.name,
        DeltaStepsEmbedding.name,
        BayesSimEmbedding.name,
        DynamicTimeWarpingEmbedding.name,
        RNNEmbedding.name,
    ],
    ids=["laststep", "allsteps", "deltasteps", "bayessim", "dtw", "rnn"],
)
@pytest.mark.parametrize("num_segments, len_segments", [(4, None), (None, 13)], ids=["numsegs4", "lensegs13"])
@pytest.mark.parametrize("stop_on_done", [True, False], ids=["stop", "dontstop"])
@pytest.mark.parametrize(
    "state_mask_labels, act_mask_labels",
    [(None, None), (["alpha", "theta"], ["V"])],
    ids=["alldims", "seldims"],
)
def test_sbi_embedding(
    ex_dir,
    env: SimEnv,
    embedding_name: str,
    num_segments: int,
    len_segments: int,
    stop_on_done: bool,
    state_mask_labels: Optional[List[str]],
    act_mask_labels: Optional[List[str]],
):
    pyrado.set_seed(0)

    # Reduce the number of steps to make this test run faster
    env.max_steps = 80

    # Policy
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # Define a mapping: index - domain parameter
    dp_mapping = {1: "mass_pend_pole", 2: "length_pend_pole"}

    # Time series embedding
    if embedding_name == LastStepEmbedding.name:
        embedding = LastStepEmbedding(
            env.spec,
            RolloutSamplerForSBI.get_dim_data(env.spec),
            state_mask_labels=state_mask_labels,
            act_mask_labels=act_mask_labels,
        )
    elif embedding_name == AllStepsEmbedding.name:
        embedding = AllStepsEmbedding(
            env.spec,
            RolloutSamplerForSBI.get_dim_data(env.spec),
            env.max_steps,
            downsampling_factor=3,
            state_mask_labels=state_mask_labels,
            act_mask_labels=act_mask_labels,
        )
    elif embedding_name == DeltaStepsEmbedding.name:
        embedding = DeltaStepsEmbedding(
            env.spec,
            RolloutSamplerForSBI.get_dim_data(env.spec),
            env.max_steps,
            downsampling_factor=3,
            state_mask_labels=state_mask_labels,
            act_mask_labels=act_mask_labels,
        )
    elif embedding_name == BayesSimEmbedding.name:
        embedding = BayesSimEmbedding(
            env.spec,
            RolloutSamplerForSBI.get_dim_data(env.spec),
            downsampling_factor=3,
            state_mask_labels=state_mask_labels,
            act_mask_labels=act_mask_labels,
        )
    elif embedding_name == DynamicTimeWarpingEmbedding.name:
        embedding = DynamicTimeWarpingEmbedding(
            env.spec,
            RolloutSamplerForSBI.get_dim_data(env.spec),
            downsampling_factor=3,
            state_mask_labels=state_mask_labels,
            act_mask_labels=act_mask_labels,
        )
    elif embedding_name == RNNEmbedding.name:
        embedding = RNNEmbedding(
            env.spec,
            RolloutSamplerForSBI.get_dim_data(env.spec),
            hidden_size=10,
            num_recurrent_layers=1,
            output_size=1,
            len_rollouts=env.max_steps,
            downsampling_factor=1,
            state_mask_labels=state_mask_labels,
            act_mask_labels=act_mask_labels,
        )
    else:
        raise NotImplementedError

    sampler = SimRolloutSamplerForSBI(
        env,
        policy,
        dp_mapping,
        embedding,
        num_segments,
        len_segments,
        stop_on_done,
        rollouts_real=None,
        use_rec_act=False,
    )

    # Test with 7 domain parameter sets
    data_sim = sampler(to.abs(to.randn(7, 2)))
    assert data_sim.shape == (7, embedding.dim_output)


@pytest.mark.slow
@pytest.mark.parametrize("algo_name", [NPDR.name, BayesSim.name], ids=["NPDR", "Bayessim"])
@pytest.mark.parametrize("env", ["default_qqsu"], ids=["qqsu"], indirect=True)
@pytest.mark.parametrize("num_segments, len_segments", [(4, None), (None, 13)], ids=["numsegs4", "lensegs13"])
@pytest.mark.parametrize("num_real_rollouts", [1, 2], ids=["1ro", "2ros"])
@pytest.mark.parametrize("num_sbi_rounds", [1, 2], ids=["1round", "2rounds"])
@pytest.mark.parametrize("use_rec_act", [True, False], ids=["userecact", "dontuserecact"])
def test_npdr_and_bayessim(
    ex_dir,
    algo_name: str,
    env: SimEnv,
    num_segments: int,
    len_segments: int,
    num_real_rollouts: int,
    num_sbi_rounds: int,
    use_rec_act: bool,
):
    pyrado.set_seed(0)

    # Create a fake ground truth target domain
    env_real = deepcopy(env)
    dp_nom = env.get_nominal_domain_param()
    env_real.domain_param = dict(
        mass_pend_pole=dp_nom["mass_pend_pole"] * 1.2, length_pend_pole=dp_nom["length_pend_pole"] * 0.8
    )

    # Reduce the number of steps to make this test run faster
    env.max_steps = 40
    env_real.max_steps = 40

    # Policy
    policy = QQubeSwingUpAndBalanceCtrl(env.spec)

    # Define a mapping: index - domain parameter
    dp_mapping = {1: "mass_pend_pole", 2: "length_pend_pole"}

    # Prior
    prior_hparam = dict(
        low=to.tensor([dp_nom["mass_pend_pole"] * 0.5, dp_nom["length_pend_pole"] * 0.5]),
        high=to.tensor([dp_nom["mass_pend_pole"] * 1.5, dp_nom["length_pend_pole"] * 1.5]),
    )
    prior = sbiutils.BoxUniform(**prior_hparam)

    # Time series embedding
    embedding = BayesSimEmbedding(
        env.spec,
        RolloutSamplerForSBI.get_dim_data(env.spec),
        downsampling_factor=3,
    )

    # Posterior (normalizing flow)
    posterior_hparam = dict(model="maf", embedding_net=nn.Identity(), hidden_features=20, num_transforms=3)

    # Policy optimization subroutine
    subrtn_policy_hparam = dict(
        max_iter=1,
        pop_size=2,
        num_init_states_per_domain=1,
        num_domains=2,
        expl_std_init=0.1,
        expl_factor=1.1,
        num_workers=1,
    )
    subrtn_policy = HCNormal(ex_dir, env, policy, **subrtn_policy_hparam)

    # Algorithm
    algo_hparam = dict(
        max_iter=1,
        num_sim_per_round=20,
        num_real_rollouts=num_real_rollouts,
        num_sbi_rounds=num_sbi_rounds,
        simulation_batch_size=1,
        normalize_posterior=False,
        num_eval_samples=2,
        num_segments=num_segments,
        len_segments=len_segments,
        use_rec_act=use_rec_act,
        stop_on_done=True,
        subrtn_sbi_training_hparam=dict(max_num_epochs=1),  # only train for 1 iteration
        # subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=True, mcmc_parameters=dict(warmup_steps=20)),
        num_workers=1,
    )
    skip = False
    if algo_name == NPDR.name:
        algo = NPDR(
            save_dir=ex_dir,
            env_sim=env,
            env_real=env_real,
            policy=policy,
            dp_mapping=dp_mapping,
            prior=prior,
            embedding=embedding,
            subrtn_sbi_class=SNPE_C,
            posterior_hparam=posterior_hparam,
            subrtn_policy=subrtn_policy,
            **algo_hparam,
        )
    elif algo_name == BayesSim.name:
        # We are not checking multi-round SNPE-A since it has known issues
        if algo_hparam["num_sbi_rounds"] > 1:
            skip = True
        algo = BayesSim(
            save_dir=ex_dir,
            env_sim=env,
            env_real=env_real,
            policy=policy,
            dp_mapping=dp_mapping,
            embedding=embedding,
            prior=prior,
            subrtn_policy=subrtn_policy,
            **algo_hparam,
        )
    else:
        raise NotImplementedError

    if not skip:
        algo.train()
        # Just checking the interface here
        assert algo.curr_iter == algo.max_iter

        # Save and load
        algo.save_snapshot(meta_info=None)
        algo_loaded = pyrado.load("algo.pkl", ex_dir)
        MockArgsSBI = namedtuple("MockArgs", "dir, policy_name, iter, round")
        args = MockArgsSBI(ex_dir, "policy", 0, num_sbi_rounds - 1)
        algo_loaded.load_snapshot(args)
        assert isinstance(algo_loaded, Algorithm)


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_qqsu"], indirect=True)
@pytest.mark.parametrize("optimize_mean", [False, True])
def test_sprl(ex_dir, env: SimEnv, optimize_mean: bool):
    pyrado.set_seed(0)

    env = ActNormWrapper(env)
    env_sprl_params = [
        dict(
            name="gravity_const",
            target_mean=to.tensor([9.81]),
            target_cov_chol_flat=to.tensor([1.0]),
            init_mean=to.tensor([9.81]),
            init_cov_chol_flat=to.tensor([0.05]),
        )
    ]
    radnomizer = DomainRandomizer(*[SelfPacedDomainParam(**p) for p in env_sprl_params])
    env = DomainRandWrapperLive(env, randomizer=radnomizer)

    policy = FNNPolicy(env.spec, hidden_sizes=[64, 64], hidden_nonlin=to.tanh)

    vfcn_hparam = dict(hidden_sizes=[32, 32], hidden_nonlin=to.relu)
    vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
    critic_hparam = dict(
        gamma=0.9844534412010116,
        lamda=0.9710614403461155,
        num_epoch=10,
        batch_size=150,
        standardize_adv=False,
        lr=0.00016985313083236645,
    )
    critic = GAE(vfcn, **critic_hparam)

    subrtn_hparam = dict(
        max_iter=1,
        eps_clip=0.12648736789309026,
        min_steps=10 * env.max_steps,
        num_epoch=3,
        batch_size=150,
        std_init=0.7573286998997557,
        lr=6.999956625305722e-04,
        max_grad_norm=1.0,
        num_workers=1,
    )

    algo_hparam = dict(
        kl_constraints_ub=8000,
        performance_lower_bound=500,
        std_lower_bound=0.4,
        kl_threshold=200,
        max_iter=1,
        optimize_mean=optimize_mean,
    )

    algo = SPRL(env, PPO(ex_dir, env, policy, critic, **subrtn_hparam), **algo_hparam)
    algo.train(snapshot_mode="latest")
    assert algo.curr_iter == algo.max_iter

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)


@pytest.mark.slow
@pytest.mark.parametrize("env", ["default_qqsu"], ids=["qq-su"], indirect=True)
@pytest.mark.parametrize("policy", ["fnn_policy"], ids=["fnn"], indirect=True)
@pytest.mark.parametrize("algo_hparam", [dict(max_iter=2, num_teachers=2)], ids=["casual"])
def test_pddr(ex_dir, env: SimEnv, policy, algo_hparam):
    pyrado.set_seed(0)

    # Create algorithm and train
    teacher_policy = deepcopy(policy)
    critic = GAE(
        vfcn=FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    )
    teacher_algo_hparam = dict(critic=critic, min_steps=1500, max_iter=2)
    teacher_algo = PPO

    # Wrapper
    randomizer = create_default_randomizer(env)
    env = DomainRandWrapperLive(env, randomizer)

    # Subroutine
    algo_hparam = dict(
        max_iter=2,
        min_steps=env.max_steps,
        std_init=0.15,
        num_epochs=10,
        num_teachers=2,
        teacher_policy=teacher_policy,
        teacher_algo=teacher_algo,
        teacher_algo_hparam=teacher_algo_hparam,
        num_workers=1,
    )

    algo = PDDR(ex_dir, env, policy, **algo_hparam)

    algo.train()

    assert algo.curr_iter == algo.max_iter

    # Save and load
    algo.save_snapshot(meta_info=None)
    algo_loaded = pyrado.load("algo.pkl", ex_dir)
    args = MockArgs(ex_dir, "policy", "vfcn")
    algo_loaded.load_snapshot(args)
    assert isinstance(algo_loaded, Algorithm)

    # Load the experiment. Since we did not save any hyper-parameters, we ignore the errors when loading.
    env, policy, extra = load_experiment(ex_dir)
    assert isinstance(env, Env)
    assert isinstance(policy, Policy)
    assert isinstance(extra, dict)
