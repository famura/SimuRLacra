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

from pyrado.algorithms.episodic.power import PoWER
from pyrado.algorithms.meta.bayrn import BayRn
from pyrado.algorithms.step_based.gae import GAE
from pyrado.algorithms.meta.arpl import ARPL
from pyrado.algorithms.episodic.cem import CEM
from pyrado.algorithms.step_based.ppo import PPO
from pyrado.algorithms.meta.spota import SPOTA
from pyrado.algorithms.step_based.svpg import SVPG
from pyrado.algorithms.episodic.sysid_via_episodic_rl import DomainDistrParamPolicy, SysIdViaEpisodicRL
from pyrado.domain_randomization.domain_parameter import UniformDomainParam
from pyrado.domain_randomization.domain_randomizer import DomainRandomizer
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.domain_randomization.default_randomizers import (
    create_default_randomizer,
    get_default_domain_param_map_qq,
    create_zero_var_randomizer,
)
from pyrado.environment_wrappers.domain_randomization import (
    DomainRandWrapperBuffer,
    DomainRandWrapperLive,
    MetaDomainRandWrapper,
)
from pyrado.environment_wrappers.state_augmentation import StateAugmentationWrapper
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.sim_base import SimEnv
from pyrado.logger import set_log_prefix_dir
from pyrado.policies.features import *
from pyrado.policies.feed_forward.fnn import FNNPolicy, FNN
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.policies.special.environment_specific import QQubeSwingUpAndBalanceCtrl
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sequences import *
from pyrado.spaces import ValueFunctionSpace, BoxSpace
from pyrado.utils.data_types import EnvSpec


@pytest.fixture
def ex_dir(tmpdir):
    # Fixture providing an experiment directory
    set_log_prefix_dir(tmpdir)
    return tmpdir


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


# TODO @Robin
# @pytest.mark.parametrize(
#     'env', [
#         'default_qqsu'
#     ],
#     ids=['qq-su'],
#     indirect=True
# )
# @pytest.mark.parametrize(
#     'subrtn_hparam', [dict(max_iter=2, min_rollouts=5, num_workers=1, num_epoch=4)],
#     ids=['casual']
# )
# @pytest.mark.parametrize(
#     'actor_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)],
#     ids=['casual']
# )
# @pytest.mark.parametrize(
#     'vfcn_hparam', [dict(hidden_sizes=[8, 8], hidden_nonlin=to.tanh)],
#     ids=['casual']
# )
# @pytest.mark.parametrize(
#     'critic_hparam', [dict(gamma=0.995, lamda=1., num_epoch=1, lr=1e-4, standardize_adv=False)],
#     ids=['casual']
# )
# @pytest.mark.parametrize(
#     'adr_hparam', [dict(max_iter=2, num_svpg_particles=3, num_discriminator_epoch=3, batch_size=100,
#                         num_workers=1, randomized_params=[])],
#     ids=['casual']
# )
# def test_adr(ex_dir, env, subrtn_hparam, actor_hparam, vfcn_hparam, critic_hparam, adr_hparam):
#     # Create the subroutine for the meta-algorithm
#     actor = FNNPolicy(spec=env.spec, **actor_hparam)
#     vfcn = FNNPolicy(spec=EnvSpec(env.obs_space, ValueFunctionSpace), **vfcn_hparam)
#     critic = GAE(vfcn, **critic_hparam)
#     subroutine = PPO(ex_dir, env, actor, critic, **subrtn_hparam)
#
#     # Create algorithm and train
#     particle_hparam = dict(actor=actor_hparam, vfcn=vfcn_hparam, critic=critic_hparam)
#     algo = ADR(ex_dir, env, subroutine, svpg_particle_hparam=particle_hparam, **adr_hparam)
#     algo.train()
#     assert algo.curr_iter == algo.max_iter


@pytest.mark.longtime
@pytest.mark.parametrize("env", ["default_qbb"], ids=["qbb"], indirect=True)
@pytest.mark.parametrize(
    "spota_hparam",
    [
        dict(
            max_iter=2,
            alpha=0.05,
            beta=0.01,
            nG=2,
            nJ=10,
            ntau=5,
            nc_init=1,
            nr_init=1,
            sequence_cand=sequence_add_init,
            sequence_refs=sequence_const,
            warmstart_cand=False,
            warmstart_refs=False,
            num_bs_reps=1000,
            studentized_ci=False,
        ),
    ],
    ids=["casual_hparam"],
)
def test_spota_ppo(ex_dir, env: SimEnv, spota_hparam):
    # Environment and domain randomization
    randomizer = create_default_randomizer(env)
    env = DomainRandWrapperBuffer(env, randomizer)

    # Policy and subroutines
    policy = FNNPolicy(env.spec, [16, 16], hidden_nonlin=to.tanh)
    vfcn = FNN(input_size=env.obs_space.flat_dim, output_size=1, hidden_sizes=[16, 16], hidden_nonlin=to.tanh)
    critic_hparam = dict(gamma=0.998, lamda=0.95, num_epoch=3, batch_size=64, lr=1e-3)
    critic_cand = GAE(vfcn, **critic_hparam)
    critic_refs = GAE(deepcopy(vfcn), **critic_hparam)

    subrtn_hparam_cand = dict(
        # min_rollouts=0,  # will be overwritten by SPOTA
        min_steps=0,  # will be overwritten by SPOTA
        max_iter=2,
        num_epoch=3,
        eps_clip=0.1,
        batch_size=64,
        num_workers=1,
        std_init=0.5,
        lr=1e-2,
    )
    subrtn_hparam_cand = subrtn_hparam_cand

    sr_cand = PPO(ex_dir, env, policy, critic_cand, **subrtn_hparam_cand)
    sr_refs = PPO(ex_dir, env, deepcopy(policy), critic_refs, **subrtn_hparam_cand)

    # Create algorithm and train
    algo = SPOTA(ex_dir, env, sr_cand, sr_refs, **spota_hparam)
    algo.train()


@pytest.mark.longtime
@pytest.mark.parametrize("env", ["default_qqsu"], ids=["qq"], indirect=True)
# @pytest.mark.parametrize("env_real", ["default_qqsu"], ids=["qq"], indirect=True)
@pytest.mark.parametrize(
    "bayrn_hparam",
    [
        dict(
            max_iter=2,
            acq_fc="UCB",
            acq_param=dict(beta=0.25),
            acq_restarts=500,
            acq_samples=1000,
            num_init_cand=3,
            warmstart=True,
            num_eval_rollouts_real=10,  # sim-2-sim
        ),
    ],
    ids=["casual_hparam"],
)
def test_bayrn_power(ex_dir, env: SimEnv, bayrn_hparam):
    # Environments and domain randomization
    env_real = deepcopy(env)
    env_sim = DomainRandWrapperLive(env, create_zero_var_randomizer(env))
    dp_map = get_default_domain_param_map_qq()
    env_sim = MetaDomainRandWrapper(env_sim, dp_map)
    env_real.domain_param = dict(Mp=0.024 * 1.1, Mr=0.095 * 1.1)
    env_real = wrap_like_other_env(env_real, env_sim)

    # Policy and subroutine
    policy_hparam = dict(energy_gain=0.587, ref_energy=0.827, acc_max=10.0)
    policy = QQubeSwingUpAndBalanceCtrl(env_sim.spec, **policy_hparam)
    subrtn_hparam = dict(
        max_iter=5,
        pop_size=40,
        num_rollouts=8,
        num_is_samples=10,
        expl_std_init=2.0,
        expl_std_min=0.02,
        symm_sampling=False,
        num_workers=1,
    )
    subrtn = PoWER(ex_dir, env_sim, policy, **subrtn_hparam)

    # Set the boundaries for the GP
    dp_nom = inner_env(env_sim).get_nominal_domain_param()
    ddp_space = BoxSpace(
        bound_lo=np.array([0.8 * dp_nom["Mp"], 1e-8, 0.8 * dp_nom["Mr"], 1e-8]),
        bound_up=np.array([1.2 * dp_nom["Mp"], 1e-7, 1.2 * dp_nom["Mr"], 1e-7]),
    )

    # Create algorithm and train
    algo = BayRn(ex_dir, env_sim, env_real, subrtn, ddp_space, **bayrn_hparam)
    algo.train()


@pytest.mark.parametrize("env", ["default_omo"], ids=["omo"], indirect=True)
def test_arpl(ex_dir, env: SimEnv):
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


@pytest.mark.longtime
@pytest.mark.parametrize("env, num_eval_rollouts", [("default_bob", 5)], ids=["bob"], indirect=["env"])
def test_sysidasrl(ex_dir, env: SimEnv, num_eval_rollouts):
    def eval_ddp_policy(rollouts_real):
        init_states_real = np.array([ro.rollout_info["init_state"] for ro in rollouts_real])
        rollouts_sim = []
        for i, _ in enumerate(range(num_eval_rollouts)):
            rollouts_sim.append(
                rollout(env_sim, behavior_policy, eval=True, reset_kwargs=dict(init_state=init_states_real[i, :]))
            )

        # Clip the rollouts rollouts yielding two lists of pairwise equally long rollouts
        ros_real_tr, ros_sim_tr = algo.truncate_rollouts(rollouts_real, rollouts_sim, replicate=False)
        assert len(ros_real_tr) == len(ros_sim_tr)
        assert all(
            [
                np.allclose(r.rollout_info["init_state"], s.rollout_info["init_state"])
                for r, s in zip(ros_real_tr, ros_sim_tr)
            ]
        )

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
    behavior_policy = LinearPolicy(env_sim.spec, feats=FeatureStack([identity_feat]))
    prior = DomainRandomizer(
        UniformDomainParam(name="ang_offset", mean=1 * np.pi / 180, halfspan=1 * np.pi / 180),
    )
    ddp_policy = DomainDistrParamPolicy(mapping=dp_map, trafo_mask=[False, True], prior=prior)

    # Subroutine
    subrtn_hparam = dict(
        max_iter=5,
        pop_size=40,
        num_rollouts=1,
        num_is_samples=4,
        expl_std_init=1 * np.pi / 180,
        expl_std_min=0.001,
        extra_expl_std_init=0.0,
        extra_expl_decay_iter=5,
        num_workers=1,
    )
    subrtn = CEM(ex_dir, env_sim, ddp_policy, **subrtn_hparam)

    algo_hparam = dict(
        metric=None, obs_dim_weight=np.ones(env_sim.obs_space.shape), num_rollouts_per_distr=10, num_workers=1
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
