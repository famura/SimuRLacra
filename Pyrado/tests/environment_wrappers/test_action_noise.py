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
from tests.conftest import m_needs_bullet, m_needs_vortex

import pyrado
from pyrado.environment_wrappers.action_noise import GaussianActNoiseWrapper
from pyrado.environment_wrappers.action_normalization import ActNormWrapper


@pytest.mark.wrapper
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qbb",
        pytest.param("default_bop2d_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_vx", marks=m_needs_vortex),
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
        pytest.param("default_bop5d_vx", marks=m_needs_vortex),
    ],
    ids=["bob", "qbb", "bop2d_b", "bop2d_v", "bop5d_b", "bop5d_v"],
    indirect=True,
)
def test_act_noise_simple(env):
    # Typical case with zero mean and non-zero std
    wrapped_env = GaussianActNoiseWrapper(env, noise_std=0.2 * np.ones(env.act_space.shape))
    for _ in range(3):
        # Sample some values
        rand_act = env.act_space.sample_uniform()
        wrapped_env.reset()
        obs_nom, _, _, _ = env.step(rand_act)
        obs_wrapped, _, _, _ = wrapped_env.step(rand_act)
        # Different actions can not lead to the same observation
        assert not np.all(obs_nom == obs_wrapped)

    # Unusual case with non-zero mean and zero std
    wrapped_env = GaussianActNoiseWrapper(env, noise_mean=0.1 * np.ones(env.act_space.shape))
    for _ in range(3):
        # Sample some values
        rand_act = env.act_space.sample_uniform()
        wrapped_env.reset()
        obs_nom, _, _, _ = env.step(rand_act)
        obs_wrapped, _, _, _ = wrapped_env.step(rand_act)
        # Different actions can not lead to the same observation
        assert not np.all(obs_nom == obs_wrapped)

    # General case with non-zero mean and non-zero std
    wrapped_env = GaussianActNoiseWrapper(
        env, noise_mean=0.1 * np.ones(env.act_space.shape), noise_std=0.2 * np.ones(env.act_space.shape)
    )
    for _ in range(3):
        # Sample some values
        rand_act = env.act_space.sample_uniform()
        wrapped_env.reset()
        obs_nom, _, _, _ = env.step(rand_act)
        obs_wrapped, _, _, _ = wrapped_env.step(rand_act)
        # Different actions can not lead to the same observation
        assert not np.all(obs_nom == obs_wrapped)


@pytest.mark.wrapper
@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_qbb",
    ],
    ids=["bob", "qbb"],
    indirect=True,
)
def test_order_act_noise_act_norm(env):
    # First noise wrapper then normalization wrapper
    wrapped_env_noise = GaussianActNoiseWrapper(
        env, noise_mean=0.2 * np.ones(env.act_space.shape), noise_std=0.1 * np.ones(env.act_space.shape)
    )
    wrapped_env_noise_norm = ActNormWrapper(wrapped_env_noise)

    # First normalization wrapper then noise wrapper
    wrapped_env_norm = ActNormWrapper(env)
    wrapped_env_norm_noise = GaussianActNoiseWrapper(
        wrapped_env_norm, noise_mean=0.2 * np.ones(env.act_space.shape), noise_std=0.1 * np.ones(env.act_space.shape)
    )

    # Sample some values directly from the act_spaces
    for i in range(3):
        pyrado.set_seed(i)
        act_noise_norm = wrapped_env_noise_norm.act_space.sample_uniform()

        pyrado.set_seed(i)
        act_norm_noise = wrapped_env_norm_noise.act_space.sample_uniform()

        # These samples must be the same since were not passed to _process_act function
        assert np.all(act_noise_norm == act_norm_noise)

    # Process a sampled action
    for i in range(3):
        # Sample a small random action such that the denormalization doe not map it to the act_space limits
        rand_act = 0.01 * env.act_space.sample_uniform()

        pyrado.set_seed(i)
        o1 = wrapped_env_noise_norm.reset()
        obs_noise_norm, _, _, _ = wrapped_env_noise_norm.step(rand_act)

        pyrado.set_seed(i)
        o2 = wrapped_env_norm_noise.reset()
        obs_norm_noise, _, _, _ = wrapped_env_norm_noise.step(rand_act)

        # The order of processing (first normalization or first randomization must make a difference)
        assert not np.all(obs_noise_norm == obs_norm_noise)
