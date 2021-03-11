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
import numpy as np
import pandas as pd
import torch as to
from copy import deepcopy
from matplotlib import pyplot as plt
from sbi.inference import simulate_for_sbi, SNPE
from sbi.user_input.user_input_checks import prepare_for_sbi
from sbi.utils import posterior_nn, BoxUniform

from pyrado.environments.sim_base import SimEnv
from pyrado.plotting.categorical import draw_categorical
from pyrado.plotting.curve import draw_curve_from_data, draw_dts
from pyrado.plotting.distribution import draw_posterior_distr_pairwise
from pyrado.plotting.rollout_based import (
    plot_observations_actions_rewards,
    plot_observations,
    plot_actions,
    plot_rewards,
    plot_potentials,
    plot_features,
)
from pyrado.plotting.surface import draw_surface
from pyrado.policies.base import Policy
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
from pyrado.sampling.rollout import rollout
from pyrado.spaces.singular import SingularStateSpace
from pyrado.utils.functions import rosenbrock


@pytest.mark.parametrize(
    "x, y, data_format",
    [
        (np.linspace(-2, 2, 30, True), np.linspace(-1, 3, 30, True), "numpy"),
        (np.linspace(-2, 2, 30, True), np.linspace(-1, 3, 30, True), "torch"),
    ],
    ids=["numpy", "torch"],
)
def test_surface(x, y, data_format):
    draw_surface(x, y, rosenbrock, "x", "y", "z", data_format)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(np.random.randn(20, 4), columns=list("ABCD")),
        np.random.randn(20, 4),
        np.random.randn(20, 4).tolist(),
    ],
    ids=["dataframe", "array", "list"],
)
def test_render_categorical(data):
    fix, axs = plt.subplots(nrows=2, ncols=2)
    draw_categorical(
        "box",
        axs[0, 0],
        data,
        x_label=["A", "b", "C", "d"],
        y_label="y",
        vline_level=None,
        palette=None,
        title=None,
        show_legend=True,
    )
    draw_categorical(
        "violin",
        axs[0, 1],
        data,
        x_label=None,
        y_label=None,
        vline_level=None,
        palette=None,
        title=None,
        show_legend=True,
    )
    draw_categorical(
        "box",
        axs[1, 0],
        data,
        x_label=["A", "b", "C", "d"],
        y_label=r"$y$",
        vline_level=None,
        palette=None,
        title="Title",
        show_legend=True,
        plot_kwargs=dict(showfliers=False),
    )
    draw_categorical(
        "violin",
        axs[1, 1],
        data,
        x_label=["A", "b", "C", "d"],
        y_label="",
        vline_level=None,
        palette=None,
        title="Title",
        show_legend=True,
        plot_kwargs=dict(showfliers=False),
    )


@pytest.mark.parametrize(
    "data, x_grid",
    [
        (pd.DataFrame(np.random.randn(20, 14), columns=["a"] * 14), np.arange(0, 20)),
        (np.random.randn(20, 14), np.arange(0, 20)),
        (np.random.randn(20, 14).tolist(), np.arange(0, 20)),
    ],
    ids=["dataframe", "array", "list"],
)
def test_render_curve(data, x_grid):
    fix, axs = plt.subplots(nrows=3, ncols=2)
    draw_curve_from_data(
        "mean_std",
        axs[0, 0],
        data,
        x_grid,
        ax_calc=1,
        x_label="A",
        y_label="y",
        vline_level=None,
        title=None,
        show_legend=True,
        area_label="a",
        plot_kwargs=dict(alpha=0.1, color="r", ls="--"),
    )
    draw_curve_from_data(
        "min_mean_max",
        axs[0, 1],
        data,
        x_grid,
        ax_calc=1,
        x_label=None,
        y_label=None,
        vline_level=None,
        title=None,
        show_legend=True,
        curve_label="c",
    )
    draw_curve_from_data(
        "mean_std",
        axs[1, 0],
        data,
        x_grid,
        ax_calc=1,
        x_label="d",
        y_label=r"$y$",
        vline_level=None,
        title="Title",
        show_legend=True,
        curve_label="c",
        plot_kwargs=dict(alpha=0.1),
    )
    draw_curve_from_data(
        "min_mean_max",
        axs[1, 1],
        data,
        x_grid,
        ax_calc=1,
        x_label=r"$\mu$",
        y_label="",
        vline_level=None,
        title="Title",
        show_legend=True,
        area_label="a",
        plot_kwargs=dict(alpha=0.1, color="r"),
    )
    draw_curve_from_data(
        "ci_on_mean",
        axs[2, 0],
        data,
        x_grid,
        ax_calc=1,
        title=None,
        show_legend=True,
        curve_label="100 reps",
        area_label="ci",
        cmp_kwargs=dict(num_reps=100, confidence_level=0.5),
        plot_kwargs=dict(color="c"),
    )
    draw_curve_from_data(
        "ci_on_mean",
        axs[2, 1],
        data,
        x_grid,
        ax_calc=1,
        title=None,
        show_legend=True,
        curve_label="10000 reps",
        area_label="ci",
        cmp_kwargs=dict(num_reps=10000),
        plot_kwargs=dict(alpha=0.4, color="c"),
    )


@pytest.mark.parametrize(
    "env, policy",
    [("default_qbb", "dummy_policy"), ("default_qbb", "linear_policy"), ("default_qbb", "nf_policy")],
    indirect=True,
)
def test_rollout_based(env, policy):
    ro = rollout(env, policy, record_dts=True)

    if isinstance(policy, LinearPolicy):
        plot_features(ro, policy)
    elif isinstance(policy, PotentialBasedPolicy):
        plot_potentials(ro)
    else:
        plot_observations_actions_rewards(ro)
        plot_observations(ro)
        plot_actions(ro, env)
        plot_rewards(ro)
        draw_dts(ro.dts_policy, ro.dts_step, ro.dts_remainder, y_top_lim=5)


@pytest.mark.parametrize(
    "env, policy",
    [("default_omo", "idle_policy")],
    indirect=True,
)
@pytest.mark.parametrize("layout", ["inside", "outside"], ids=["inside", "outside"])
@pytest.mark.parametrize("x_labels, y_labels, prob_labels", [(None, None, None), ("", "", "")], ids=["None", "default"])
def test_pair_plot(env: SimEnv, policy: Policy, layout: str, x_labels, y_labels, prob_labels):
    def _simulator(dp: to.Tensor) -> to.Tensor:
        """ The most simple interface of a simulation to sbi, using `env` and `policy` from outer scope """
        ro = rollout(env, policy, eval=True, reset_kwargs=dict(domain_param=dict(m=dp[0], k=dp[1], d=dp[2])))
        observation_sim = to.from_numpy(ro.observations[-1]).to(dtype=to.float32)
        return to.atleast_2d(observation_sim)

    # Fix the init state
    env.init_space = SingularStateSpace(env.init_space.sample_uniform())
    env_real = deepcopy(env)
    env_real.domain_param = {"m": 0.8, "k": 35, "d": 0.7}

    # Domain parameter mapping and prior
    dp_mapping = {0: "m", 1: "k", 2: "d"}
    prior = BoxUniform(low=to.tensor([0.5, 20, 0.2]), high=to.tensor([1.5, 40, 0.8]))

    # Learn a likelihood from the simulator
    density_estimator = posterior_nn(model="maf", hidden_features=10, num_transforms=3)
    snpe = SNPE(prior, density_estimator)
    simulator, prior = prepare_for_sbi(_simulator, prior)
    domain_param, data_sim = simulate_for_sbi(
        simulator=simulator,
        proposal=prior,
        num_simulations=50,
        num_workers=4,
    )
    snpe.append_simulations(domain_param, data_sim)
    density_estimator = snpe.train(max_num_epochs=5)
    posterior = snpe.build_posterior(density_estimator)

    # Create a fake (random) true domain parameter
    domain_param_gt = to.tensor([env_real.domain_param[key] for _, key in dp_mapping.items()])
    domain_param_gt += domain_param_gt * to.randn(len(dp_mapping)) / 5
    domain_param_gt = domain_param_gt.unsqueeze(0)
    data_real = simulator(domain_param_gt)

    # Get a (random) condition
    condition = domain_param_gt.clone()

    if layout == "inside":
        num_rows, num_cols = len(dp_mapping), len(dp_mapping)
    else:
        num_rows, num_cols = len(dp_mapping) + 1, len(dp_mapping) + 1

    _, axs = plt.subplots(num_rows, num_cols, figsize=(14, 14), tight_layout=True)
    fig = draw_posterior_distr_pairwise(
        axs,
        posterior,
        data_real,
        dp_mapping,
        condition,
        prior,
        env_real,
        marginal_layout=layout,
        grid_res=100,
        normalize_posterior=False,
        rescale_posterior=True,
        x_labels=x_labels,
        y_labels=y_labels,
        prob_labels=prob_labels,
    )

    assert fig is not None
