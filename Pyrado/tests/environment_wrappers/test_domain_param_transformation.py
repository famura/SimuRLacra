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

import random
from typing import Type

import pytest
from tests.conftest import VORTEX_ONLY_DOMAIN_PARAM_LIST, m_needs_bullet, m_needs_mujoco, m_needs_vortex

import pyrado
from pyrado.domain_randomization.transformations import (
    DomainParamTransform,
    LogDomainParamTransform,
    SqrtDomainParamTransform,
)
from pyrado.environment_wrappers.utils import inner_env
from pyrado.environments.pysim.base import SimPyEnv
from pyrado.environments.sim_base import SimEnv


@pytest.mark.wrapper
@pytest.mark.parametrize(
    "env",
    [
        "default_bobd",
        "default_bob",
        "default_omo",
        "default_pend",
        "default_qbb",
        "default_qqst",
        "default_qqsu",
        "default_qcpst",
        "default_qcpsu",
        pytest.param("default_p3l_ika_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_bt", marks=m_needs_bullet),
        pytest.param("default_p3l_ta_vx", marks=m_needs_vortex),
        pytest.param("default_bop2d_bt", marks=m_needs_bullet),
        pytest.param("default_bop2d_vx", marks=m_needs_vortex),
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
        pytest.param("default_bop5d_vx", marks=m_needs_vortex),
        pytest.param("default_bs_ds_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bs_ds_pos_vx", marks=m_needs_vortex),
        pytest.param("default_bit_ika_pos_bt", marks=m_needs_bullet),
        pytest.param("default_bit_ds_vel_bt", marks=m_needs_bullet),
        pytest.param("default_cth", marks=m_needs_mujoco),
        pytest.param("default_hop", marks=m_needs_mujoco),
        pytest.param("default_wambic", marks=m_needs_mujoco),
    ],
    indirect=True,
)
@pytest.mark.parametrize("trafo_class", [LogDomainParamTransform, SqrtDomainParamTransform], ids=["log", "sqrt"])
def test_domain_param_transforms(env: SimEnv, trafo_class: Type):
    pyrado.set_seed(0)

    # Create a mask for a random domain parameter
    offset = 1
    idx = random.randint(0, len(env.supported_domain_param) - 1)
    sel_dp_change = list(env.supported_domain_param)[idx]
    sel_dp_fix = list(env.supported_domain_param)[(idx + offset) % len(env.supported_domain_param)]
    while (
        offset == 1
        or any([item in sel_dp_change for item in VORTEX_ONLY_DOMAIN_PARAM_LIST])
        or any([item in sel_dp_fix for item in VORTEX_ONLY_DOMAIN_PARAM_LIST])
    ):
        idx = random.randint(0, len(env.supported_domain_param) - 1)
        sel_dp_change = list(env.supported_domain_param)[idx]
        sel_dp_fix = list(env.supported_domain_param)[(idx + offset) % len(env.supported_domain_param)]
        offset += 1

    mask = (sel_dp_change,)
    wenv = trafo_class(env, mask)
    assert isinstance(wenv, DomainParamTransform)

    # Check 5 random values
    for _ in range(5):
        # Change the selected domain parameter
        new_dp_val = random.random() * env.get_nominal_domain_param()[sel_dp_change]
        new_dp_val = abs(new_dp_val) + 1e-6  # due to the domain of the new params
        transformed_new_dp_val = wenv.forward(new_dp_val)
        wenv.domain_param = {sel_dp_change: transformed_new_dp_val}  # calls inverse transform
        if not isinstance(inner_env(wenv), SimPyEnv):
            wenv.reset()  # the RcsPySim and MujocoSim classes need to be reset to apply the new domain param

        # Test the actual domain param and the the getters
        assert inner_env(wenv)._domain_param[sel_dp_change] == pytest.approx(new_dp_val, abs=1e-5)
        assert wenv.domain_param[sel_dp_change] == pytest.approx(new_dp_val, abs=1e-5)
        assert wenv.domain_param[sel_dp_fix] != pytest.approx(new_dp_val)
