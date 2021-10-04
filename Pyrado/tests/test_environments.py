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

import numpy as np
import pytest
import torch as to
from tests.conftest import m_needs_bullet, m_needs_mujoco, m_needs_vortex

from pyrado.environments.pysim.quanser_ball_balancer import QBallBalancerKin, QBallBalancerSim
from pyrado.environments.real_base import RealEnv
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.feed_forward.dummy import DummyPolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


@pytest.mark.parametrize(
    "env",
    [
        "default_cata",
        "default_rosen",
        "default_bobd",
        "default_bob",
        "default_omo",
        "default_pend",
        "default_qbb",
        "default_qqst",
        "default_qqsu",
        "default_qcpst",
        "default_qcpsu",
        pytest.param("default_qqst_mj", marks=m_needs_mujoco),
        pytest.param("default_qqsu_mj", marks=m_needs_mujoco),
        pytest.param("default_qqsurcs_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ika_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ika_5l_bt", marks=m_needs_bullet),
        pytest.param("default_pi_ika_6l_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ta_6l_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_vx", marks=m_needs_vortex),
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
        pytest.param("default_bop5d_vx", marks=m_needs_vortex),
        pytest.param("default_bs_ds_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bs_ds_pos_vx", marks=m_needs_vortex),
        pytest.param("default_bit_ika_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bit_ds_vel_bt", marks=m_needs_bullet),
        pytest.param("default_mg_ik_bt", marks=m_needs_bullet),
        pytest.param("default_mg_jnt_bt", marks=m_needs_bullet),
        pytest.param("default_cth", marks=m_needs_mujoco),
        pytest.param("default_hop", marks=m_needs_mujoco),
        pytest.param("default_hum", marks=m_needs_mujoco),
        pytest.param("default_ant", marks=m_needs_mujoco),
        pytest.param("default_wambic", marks=m_needs_mujoco),
    ],
    indirect=True,
)
def test_rollout(env):
    assert isinstance(env, SimEnv)

    # Hand coded rollout
    env.reset()
    done = False
    while not done:
        state, rew, done, info = env.step(0.1 * env.act_space.sample_uniform())
    assert env.curr_step <= env.max_steps

    # Rollout function
    policy = DummyPolicy(env.spec)
    ro = rollout(env, policy, eval=True)
    assert ro.length <= env.max_steps


@pytest.mark.parametrize(
    "env",
    [
        "default_cata",
        "default_rosen",
        "default_bobd",
        "default_bob",
        "default_omo",
        "default_pend",
        "default_qbb",
        "default_qqst",
        "default_qqsu",
        "default_qcpst",
        "default_qcpsu",
        pytest.param("default_qqst_mj", marks=m_needs_mujoco),
        pytest.param("default_qqsu_mj", marks=m_needs_mujoco),
        pytest.param("default_qqsurcs_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ika_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ika_5l_bt", marks=m_needs_bullet),
        pytest.param("default_pi_ika_6l_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ta_6l_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_vx", marks=m_needs_vortex),
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
        pytest.param("default_bop5d_vx", marks=m_needs_vortex),
        pytest.param("default_bs_ds_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bs_ds_pos_vx", marks=m_needs_vortex),
        pytest.param("default_bit_ika_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bit_ds_vel_bt", marks=m_needs_bullet),
        pytest.param("default_mg_ik_bt", marks=m_needs_bullet),
        pytest.param("default_mg_jnt_bt", marks=m_needs_bullet),
        pytest.param("default_cth", marks=m_needs_mujoco),
        pytest.param("default_hop", marks=m_needs_mujoco),
        pytest.param("default_hum", marks=m_needs_mujoco),
        pytest.param("default_ant", marks=m_needs_mujoco),
        pytest.param("default_wambic", marks=m_needs_mujoco),
    ],
    indirect=True,
)
def test_init_spaces(env):
    assert isinstance(env, SimEnv)
    # Test using 100 random samples per environment
    for _ in range(100):
        init_space_sample = env.init_space.sample_uniform()
        assert env.init_space.contains(init_space_sample)
        init_obs = env.reset(init_space_sample)
        assert env.obs_space.contains(init_obs)
        assert env.state_space.contains(env.state)


@pytest.mark.parametrize(
    "env",
    [
        "default_cata",
        "default_rosen",
        "default_bobd",
        "default_bob",
        "default_omo",
        "default_pend",
        "default_qbb",
        "default_qqst",
        "default_qqsu",
        "default_qcpst",
        "default_qcpsu",
        pytest.param("default_qqst_mj", marks=m_needs_mujoco),
        pytest.param("default_qqsu_mj", marks=m_needs_mujoco),
        pytest.param("default_qqsurcs_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ika_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ika_5l_bt", marks=m_needs_bullet),
        pytest.param("default_pi_ika_6l_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ta_6l_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_vx", marks=m_needs_vortex),
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
        pytest.param("default_bop5d_vx", marks=m_needs_vortex),
        pytest.param("default_bs_ds_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bs_ds_pos_vx", marks=m_needs_vortex),
        pytest.param("default_bit_ika_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bit_ds_vel_bt", marks=m_needs_bullet),
        pytest.param("default_mg_ik_bt", marks=m_needs_bullet),
        pytest.param("default_mg_jnt_bt", marks=m_needs_bullet),
        pytest.param("default_cth", marks=m_needs_mujoco),
        pytest.param("default_hop", marks=m_needs_mujoco),
        pytest.param("default_hum", marks=m_needs_mujoco),
        pytest.param("default_ant", marks=m_needs_mujoco),
        pytest.param("default_wambic", marks=m_needs_mujoco),
    ],
    indirect=True,
)
def test_reset(env):
    assert isinstance(env, SimEnv)
    # Test using 50 random samples per environment
    for _ in range(50):
        # Reset the env to a random state
        env.reset()
        env.render(mode=RenderMode(text=True))
        assert env.state_space.contains(env.state, verbose=True)

    # Reset with explicitly specified init state
    init_state = env.init_space.sample_uniform()

    # Explicitly specify once
    obs1 = env.reset(init_state=init_state)
    env.render(mode=RenderMode(text=True))
    assert env.state_space.contains(env.state, verbose=True)

    # Reset to a random state
    env.reset()

    # Reset to fixed state again
    obs2 = env.reset(init_state=init_state)
    assert obs2 == pytest.approx(obs1)


@pytest.mark.visual
@pytest.mark.parametrize(
    "env",
    [
        "default_bobd",
        "default_bob",
        "default_omo",
        "default_pend",
        "default_qbb",
        "default_qqsu",
        "default_qqst",
        "default_qcpsu",
        "default_qcpst",
    ],
    indirect=True,
)
@pytest.mark.parametrize("use_render", [False], ids=["render_off"])
def test_panda3d_animations(env, use_render):
    assert isinstance(env, SimEnv)
    env.reset()
    env.render(mode=RenderMode(video=True, render=use_render))
    for _ in range(300):  # do max 300 steps
        state, rew, done, info = env.step(np.ones(env.act_space.shape))
        env.render(mode=RenderMode(video=True, render=use_render))
        if done:
            break
    env.reset()  # only calls _reset_anim if window is already existing
    assert env.curr_step <= env.max_steps
    env._visual.destroy()
    del env._visual


@pytest.mark.visual
@pytest.mark.parametrize(
    "env",
    [
        pytest.param("default_qqsurcs_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ika_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ika_5l_bt", marks=m_needs_bullet),
        pytest.param("default_pi_ika_6l_vx", marks=m_needs_vortex),
        pytest.param("default_pi_ta_6l_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_vx", marks=m_needs_vortex),
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
        pytest.param("default_bop5d_vx", marks=m_needs_vortex),
        pytest.param("default_bs_ds_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bs_ds_pos_vx", marks=m_needs_vortex),
        pytest.param("default_bit_ika_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bit_ds_vel_bt", marks=m_needs_bullet),
        pytest.param("default_mg_ik_bt", marks=m_needs_bullet),
        pytest.param("default_mg_jnt_bt", marks=m_needs_bullet),
    ],
    indirect=True,
)
def test_rcspysim_animations(env):
    assert isinstance(env, SimEnv)
    env.reset()
    env.render(mode=RenderMode(video=True))
    for _ in range(300):  # do max 300 steps
        state, rew, done, info = env.step(np.ones(env.act_space.shape))
        env.render(mode=RenderMode(video=True))
        if done:
            break
    assert env.curr_step <= env.max_steps


@m_needs_mujoco
@pytest.mark.visual
@pytest.mark.parametrize(
    "env", ["default_cth", "default_hop", "default_hum", "default_ant", "default_wambic"], indirect=True
)
def test_mujoco_animations(env):
    assert isinstance(env, SimEnv)
    env.reset()
    env.render(mode=RenderMode(video=True))
    for _ in range(300):  # do max 300 steps
        state, rew, done, info = env.step(np.ones(env.act_space.shape))
        env.render(mode=RenderMode(video=True))
        if done:
            break
    assert env.curr_step <= env.max_steps


@pytest.mark.parametrize(
    "env",
    [
        "default_qqsu",
    ],
    indirect=True,
)
def test_gym_wrapper(env):
    gym = pytest.importorskip("gym")

    gym_env = gym.make("SimulacraPySimEnv-v0", env=env)
    gym_env.reset()
    act = gym_env.action_space.sample()
    out = gym_env.step(act)
    for el in out:
        assert el is not None


@pytest.mark.parametrize(
    "servo_ang", [np.r_[np.linspace(-np.pi / 2.1, np.pi / 2.1), np.linspace(np.pi / 2.1, -np.pi / 2.1)]]
)
def test_qbb_kin(servo_ang):
    env = QBallBalancerSim(dt=0.02, max_steps=100)
    kin = QBallBalancerKin(env, num_opt_iter=50, render_mode=RenderMode(video=False))

    servo_ang = to.tensor(servo_ang, dtype=to.get_default_dtype())
    for th in servo_ang:
        plate_ang = kin(th)
        assert plate_ang is not None


@pytest.mark.parametrize(
    "env", ["default_qqbb_real", "default_qcpst_real", "default_qcpsu_real", "default_qq_real"], indirect=True
)
def test_quanser_real_wo_connecting(env: RealEnv):
    assert env is not None
    env.render(RenderMode(text=True))


@pytest.mark.visual
@pytest.mark.parametrize(
    "env_name",
    ["MountainCar-v0", "CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0", "Pendulum-v0", "LunarLander-v2"],
)
def test_gym_env(env_name):
    # Checking the classic control problems
    gym_module = pytest.importorskip("pyrado.environments.pysim.openai_classical_control")

    env = gym_module.GymEnv(env_name)
    assert env is not None
    env.reset()
    for _ in range(50):
        env.render(RenderMode(video=True))
        act = env.act_space.sample_uniform()
        env.step(act)
    env.close()
