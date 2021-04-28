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

import numpy as np
import pytest

from pyrado.domain_randomization.default_randomizers import create_default_randomizer
from pyrado.domain_randomization.transformations import LogDomainParamTransform
from pyrado.domain_randomization.utils import wrap_like_other_env
from pyrado.environment_wrappers.action_delay import ActDelayWrapper
from pyrado.environment_wrappers.action_noise import GaussianActNoiseWrapper
from pyrado.environment_wrappers.action_normalization import ActNormWrapper
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.downsampling import DownsamplingWrapper
from pyrado.environment_wrappers.observation_noise import GaussianObsNoiseWrapper
from pyrado.environment_wrappers.observation_normalization import ObsNormWrapper, ObsRunningNormWrapper
from pyrado.environment_wrappers.observation_partial import ObsPartialWrapper
from pyrado.environment_wrappers.utils import inner_env, remove_env, typed_env
from pyrado.environments.pysim.quanser_cartpole import QCartPoleSwingUpSim
from pyrado.environments.sim_base import SimEnv
from pyrado.policies.feed_forward.dummy import DummyPolicy, IdlePolicy
from pyrado.sampling.rollout import rollout
from pyrado.utils.data_types import RenderMode


@pytest.mark.wrapper
@pytest.mark.parametrize("env", ["default_bob"], indirect=True)
def test_combination_wrappers_domain_params(env: SimEnv):
    env_d = DownsamplingWrapper(env, factor=5)
    env_do = GaussianObsNoiseWrapper(
        env_d, noise_std=2 * np.ones(env_d.obs_space.shape), noise_mean=3 * np.ones(env_d.obs_space.shape)
    )
    env_dot = LogDomainParamTransform(env_do, mask=list(env_do.supported_domain_param))

    assert env_dot.domain_param["downsampling"] == 5
    assert np.all(env_dot.domain_param["obs_noise_std"] == 2 * np.ones(env_d.obs_space.shape))
    assert np.all(env_dot.domain_param["obs_noise_mean"] == 3 * np.ones(env_d.obs_space.shape))


@pytest.mark.wrapper
def test_combination():
    env = QCartPoleSwingUpSim(dt=1 / 100.0, max_steps=20)

    randomizer = create_default_randomizer(env)
    env_r = DomainRandWrapperBuffer(env, randomizer)
    env_r.fill_buffer(num_domains=3)

    dp_before = []
    dp_after = []
    for i in range(4):
        dp_before.append(env_r.domain_param)
        rollout(env_r, DummyPolicy(env_r.spec), eval=True, seed=0, render_mode=RenderMode())
        dp_after.append(env_r.domain_param)
        assert dp_after[i] != dp_before[i]
    assert dp_after[0] == dp_after[3]

    env_rn = ActNormWrapper(env)
    elb = {"x_dot": -213.0, "theta_dot": -42.0}
    eub = {"x_dot": 213.0, "theta_dot": 42.0, "x": 0.123}
    env_rn = ObsNormWrapper(env_rn, explicit_lb=elb, explicit_ub=eub)
    alb, aub = env_rn.act_space.bounds
    assert all(alb == -1)
    assert all(aub == 1)
    olb, oub = env_rn.obs_space.bounds
    assert all(olb == -1)
    assert all(oub == 1)

    ro_r = rollout(env_r, DummyPolicy(env_r.spec), eval=True, seed=0, render_mode=RenderMode())
    ro_rn = rollout(env_rn, DummyPolicy(env_rn.spec), eval=True, seed=0, render_mode=RenderMode())
    assert np.allclose(env_rn._process_obs(ro_r.observations), ro_rn.observations)

    env_rnp = ObsPartialWrapper(env_rn, idcs=["x_dot", r"cos_theta"])
    ro_rnp = rollout(env_rnp, DummyPolicy(env_rnp.spec), eval=True, seed=0, render_mode=RenderMode())

    env_rnpa = GaussianActNoiseWrapper(
        env_rnp, noise_mean=0.5 * np.ones(env_rnp.act_space.shape), noise_std=0.1 * np.ones(env_rnp.act_space.shape)
    )
    ro_rnpa = rollout(env_rnpa, DummyPolicy(env_rnpa.spec), eval=True, seed=0, render_mode=RenderMode())
    assert not np.allclose(ro_rnp.observations, ro_rnpa.observations)  # the action noise changed to rollout

    env_rnpd = ActDelayWrapper(env_rnp, delay=3)
    ro_rnpd = rollout(env_rnpd, DummyPolicy(env_rnpd.spec), eval=True, seed=0, render_mode=RenderMode())
    assert np.allclose(ro_rnp.actions, ro_rnpd.actions)
    assert not np.allclose(ro_rnp.observations, ro_rnpd.observations)

    assert isinstance(inner_env(env_rnpd), QCartPoleSwingUpSim)
    assert typed_env(env_rnpd, ObsPartialWrapper) is not None
    assert isinstance(env_rnpd, ActDelayWrapper)
    env_rnpdr = remove_env(env_rnpd, ActDelayWrapper)
    assert not isinstance(env_rnpdr, ActDelayWrapper)


@pytest.mark.wrapper
@pytest.mark.parametrize(
    "env",
    [
        "default_qbb",
    ],
    ids=["qbb"],
    indirect=True,
)
def test_wrap_like_other_env(env: SimEnv):
    wenv_like = deepcopy(env)
    wenv_like.dt /= 3

    wenv = DownsamplingWrapper(env, factor=3)
    assert type(wenv_like) != type(wenv)
    wenv_like = wrap_like_other_env(wenv_like, wenv, use_downsampling=True)
    assert type(wenv_like) == type(wenv)

    wenv = ActNormWrapper(wenv)
    assert type(wenv_like) != type(wenv)
    wenv_like = wrap_like_other_env(wenv_like, wenv)
    assert type(wenv_like) == type(wenv)

    wenv = ObsNormWrapper(wenv)
    assert type(wenv_like) != type(wenv)
    wenv_like = wrap_like_other_env(wenv_like, wenv)
    assert type(wenv_like) == type(wenv)
    assert type(wenv_like.wrapped_env) == type(wenv.wrapped_env)

    wenv = ObsRunningNormWrapper(wenv)
    wenv_like = wrap_like_other_env(wenv_like, wenv)
    assert type(wenv_like) == type(wenv)
    assert type(wenv_like.wrapped_env) == type(wenv.wrapped_env)

    wenv = ObsPartialWrapper(wenv, idcs=["x"])
    wenv_like = wrap_like_other_env(wenv_like, wenv)
    assert type(wenv_like) == type(wenv)
    assert type(wenv_like.wrapped_env) == type(wenv.wrapped_env)
