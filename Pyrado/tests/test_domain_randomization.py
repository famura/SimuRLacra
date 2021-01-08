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
import torch as to
import numpy as np
from copy import deepcopy

import pyrado
from pyrado.domain_randomization.domain_randomizer import DistributionFreeDomainRandomizer
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperLive
from pyrado.environments.sim_base import SimEnv
from tests.conftest import m_needs_bullet, m_needs_mujoco

from pyrado.domain_randomization.domain_parameter import (
    NormalDomainParam,
    MultivariateNormalDomainParam,
    BernoulliDomainParam,
)
from pyrado.domain_randomization.utils import param_grid


@pytest.mark.parametrize(
    "dp",
    [
        NormalDomainParam(name="", mean=10, std=1.0, clip_lo=9, clip_up=11),
        MultivariateNormalDomainParam(name="", mean=to.ones((2, 1)), cov=to.eye(2), clip_lo=-1, clip_up=1.0),
        MultivariateNormalDomainParam(name="", mean=10 * to.ones((2,)), cov=2 * to.eye(2), clip_up=11),
        BernoulliDomainParam(name="", val_0=2, val_1=5, prob_1=0.8),
        BernoulliDomainParam(name="", val_0=-3, val_1=5, prob_1=0.8, clip_up=4),
    ],
    ids=["1dim", "2dim_v1", "2dim_v2", "bern_v1", "bern_v2"],
)
def test_domain_param(dp):
    for num_samples in [1, 5, 25]:
        s = dp.sample(num_samples)
        assert len(s) == num_samples


def test_randomizer(default_randomizer):
    print(default_randomizer)
    # Generate 7 samples
    default_randomizer.randomize(7)

    # Test all variations of the getter function's parameters format and dtype
    pp_3_to_dict = default_randomizer.get_params(3, fmt="dict", dtype="torch")
    assert isinstance(pp_3_to_dict, dict)
    assert isinstance(pp_3_to_dict["mass"], list)
    assert len(pp_3_to_dict["mass"]) == 3
    assert isinstance(pp_3_to_dict["mass"][0], to.Tensor)
    assert isinstance(pp_3_to_dict["multidim"][0], to.Tensor) and pp_3_to_dict["multidim"][0].shape[0] == 2
    pp_3_to_list = default_randomizer.get_params(3, fmt="list", dtype="torch")
    assert isinstance(pp_3_to_list, list)
    assert len(pp_3_to_list) == 3
    assert isinstance(pp_3_to_list[0], dict)
    assert isinstance(pp_3_to_list[0]["mass"], to.Tensor)
    assert isinstance(pp_3_to_list[0]["multidim"], to.Tensor) and pp_3_to_list[0]["multidim"].shape[0] == 2
    pp_3_np_dict = default_randomizer.get_params(3, fmt="dict", dtype="numpy")
    assert isinstance(pp_3_np_dict, dict)
    assert isinstance(pp_3_np_dict["mass"], list)
    assert len(pp_3_np_dict["mass"]) == 3
    assert isinstance(pp_3_np_dict["mass"][0], np.ndarray)
    assert isinstance(pp_3_np_dict["multidim"][0], np.ndarray) and pp_3_np_dict["multidim"][0].size == 2
    pp_3_np_list = default_randomizer.get_params(3, fmt="list", dtype="numpy")
    assert isinstance(pp_3_np_list, list)
    assert len(pp_3_np_list) == 3
    assert isinstance(pp_3_np_list[0], dict)
    assert isinstance(pp_3_np_list[0]["mass"], np.ndarray)
    assert isinstance(pp_3_np_list[0]["multidim"], np.ndarray) and pp_3_np_list[0]["multidim"].size == 2

    pp_all_to_dict = default_randomizer.get_params(-1, fmt="dict", dtype="torch")
    assert isinstance(pp_all_to_dict, dict)
    assert isinstance(pp_all_to_dict["mass"], list)
    assert len(pp_all_to_dict["mass"]) == 7
    assert isinstance(pp_all_to_dict["mass"][0], to.Tensor)
    assert isinstance(pp_all_to_dict["multidim"][0], to.Tensor) and pp_all_to_dict["multidim"][0].shape[0] == 2
    pp_all_to_list = default_randomizer.get_params(-1, fmt="list", dtype="torch")
    assert isinstance(pp_all_to_list, list)
    assert len(pp_all_to_list) == 7
    assert isinstance(pp_all_to_list[0], dict)
    assert isinstance(pp_all_to_list[0]["mass"], to.Tensor)
    assert isinstance(pp_all_to_list[0]["multidim"], to.Tensor) and pp_all_to_list[0]["multidim"].shape[0] == 2
    pp_all_np_dict = default_randomizer.get_params(-1, fmt="dict", dtype="numpy")
    assert isinstance(pp_all_np_dict, dict)
    assert isinstance(pp_all_np_dict["mass"], list)
    assert len(pp_all_np_dict["mass"]) == 7
    assert isinstance(pp_all_np_dict["mass"][0], np.ndarray)
    assert isinstance(pp_all_np_dict["multidim"][0], np.ndarray) and pp_all_np_dict["multidim"][0].size == 2
    pp_all_np_list = default_randomizer.get_params(-1, fmt="list", dtype="numpy")
    assert isinstance(pp_all_np_list, list)
    assert len(pp_all_to_list) == 7
    assert isinstance(pp_all_np_list[0], dict)
    assert isinstance(pp_all_np_list[0]["mass"], np.ndarray)
    assert isinstance(pp_all_np_list[0]["multidim"], np.ndarray) and pp_all_np_list[0]["multidim"].size == 2


def test_rescaling(default_randomizer):
    # This test relies on a fixed structure of the default_randomizer (mass is ele 0, and length is ele 2 in the list)
    randomizer = deepcopy(default_randomizer)
    randomizer.rescale_distr_param("std", 12.5)
    # Check if the right parameter of the distribution changed
    assert randomizer.domain_params[0].std == 12.5 * default_randomizer.domain_params[0].std
    assert randomizer.domain_params[2].std == 12.5 * default_randomizer.domain_params[2].std
    # Check if the other parameters were unchanged (lazily just check one attribute)
    assert randomizer.domain_params[0].mean == default_randomizer.domain_params[0].mean
    assert randomizer.domain_params[2].mean == default_randomizer.domain_params[2].mean


def test_param_grid():
    # Create a parameter grid spec
    pspec = {"p1": np.array([0.1, 0.2]), "p2": np.array([0.4, 0.5]), "p3": 3}  # fixed value

    # Create the grid entries
    pgrid = param_grid(pspec)

    # Check for presence of all entries, their order is not mandatory
    assert {"p1": 0.1, "p2": 0.4, "p3": 3} in pgrid
    assert {"p1": 0.2, "p2": 0.4, "p3": 3} in pgrid
    assert {"p1": 0.1, "p2": 0.5, "p3": 3} in pgrid
    assert {"p1": 0.2, "p2": 0.5, "p3": 3} in pgrid


@pytest.mark.parametrize(
    "env",
    [
        "default_bob",
        "default_omo",
        "default_pend",
        "default_qbb",
        "default_qcpst",
        "default_qcpsu",
        pytest.param("default_bop2d_bt", marks=m_needs_bullet),
        pytest.param("default_bop5d_bt", marks=m_needs_bullet),
        pytest.param("default_cth", marks=m_needs_mujoco),
        pytest.param("default_hop", marks=m_needs_mujoco),
        pytest.param("default_wambic", marks=m_needs_mujoco),
    ],
    ids=["bob", "omo", "pend", "qbb", "qcp-st", "qcp-su", "bop2d", "bop5d", "cth", "hop", "wam-bic"],
    indirect=True,
)
def test_setting_dp_vals(env: SimEnv):
    # Loop over all possible domain parameters and set them to a random value
    for _ in range(5):
        for dp_key in env.supported_domain_param:
            if any([s in dp_key for s in ["slip", "compliance", "linearvelocitydamping", "angularvelocitydamping"]]):
                # Skip the parameters that are only available in Vortex but not in Bullet
                assert True
            else:
                nominal_val = env.domain_param.get(dp_key)
                rand_val = nominal_val + nominal_val * np.random.rand() / 10
                env.reset(domain_param={dp_key: rand_val})
                assert env.domain_param[dp_key] == pytest.approx(rand_val, abs=5e-4)  # rolling friction is imprecise
