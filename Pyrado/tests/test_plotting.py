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
from typing import Optional

import numpy as np
import pandas as pd
import pytest
import torch as to
from matplotlib import pyplot as plt
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform, posterior_nn
from sbi.utils.user_input_checks import prepare_for_sbi

from pyrado.algorithms.meta.sbi_base import SBIBase
from pyrado.domain_randomization.transformations import LogDomainParamTransform, SqrtDomainParamTransform
from pyrado.environments.sim_base import SimEnv
from pyrado.plotting.categorical import draw_categorical
from pyrado.plotting.curve import draw_curve_from_data, draw_dts
from pyrado.plotting.distribution import draw_posterior_pairwise_heatmap, draw_posterior_pairwise_scatter
from pyrado.plotting.rollout_based import (
    plot_actions,
    plot_features,
    plot_observations,
    plot_observations_actions_rewards,
    plot_potentials,
    plot_rewards,
)
from pyrado.plotting.surface import draw_surface
from pyrado.policies.base import Policy
from pyrado.policies.feed_back.linear import LinearPolicy
from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
from pyrado.sampling.rollout import rollout
from pyrado.sampling.sbi_embeddings import Embedding
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
    [("default_bob", "dummy_policy"), ("default_qbb", "linear_policy"), ("default_qbb", "nf_policy")],
    indirect=True,
)
def test_rollout_based(env: SimEnv, policy: Policy):
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


@pytest.mark.parametrize("env, policy", [("default_omo", "idle_policy")], indirect=True)
@pytest.mark.parametrize("layout", ["inside", "outside"], ids=["inside", "outside"])
@pytest.mark.parametrize("labels", [None, ""], ids=["nolabels", "deflabels"])
@pytest.mark.parametrize("prob_labels", [None, ""], ids=["noproblabels", "defproblabels"])
@pytest.mark.parametrize("use_prior", [True, False], ids=["useprior", "dontuseprior"])
@pytest.mark.parametrize("use_trafo", [False, True], ids=["no_trafo", "trafo"])
def test_pair_plot(
    env: SimEnv,
    policy: Policy,
    layout: str,
    labels: Optional[str],
    prob_labels: Optional[str],
    use_prior: bool,
    use_trafo: bool,
):
    def _simulator(dp: to.Tensor) -> to.Tensor:
        """The most simple interface of a simulation to sbi, using `env` and `policy` from outer scope"""
        ro = rollout(env, policy, eval=True, reset_kwargs=dict(domain_param=dict(m=dp[0], k=dp[1], d=dp[2])))
        observation_sim = to.from_numpy(ro.observations[-1]).to(dtype=to.float32)
        return to.atleast_2d(observation_sim)

    # Fix the init state
    env.init_space = SingularStateSpace(env.init_space.sample_uniform())
    env_real = deepcopy(env)
    env_real.domain_param = {"m": 0.8, "k": 35, "d": 0.7}

    # Optionally transformed domain parameters for inference
    if use_trafo:
        env = SqrtDomainParamTransform(env, mask=["k"])

    # Domain parameter mapping and prior
    dp_mapping = {0: "m", 1: "k", 2: "d"}
    prior = BoxUniform(low=to.tensor([0.5, 20, 0.2]), high=to.tensor([1.5, 40, 0.8]))

    # Learn a likelihood from the simulator
    density_estimator = posterior_nn(model="maf", hidden_features=10, num_transforms=3)
    snpe = SNPE(prior, density_estimator)
    simulator, prior = prepare_for_sbi(_simulator, prior)
    domain_param, data_sim = simulate_for_sbi(simulator=simulator, proposal=prior, num_simulations=50, num_workers=1)
    snpe.append_simulations(domain_param, data_sim)
    density_estimator = snpe.train(max_num_epochs=5)
    posterior = snpe.build_posterior(density_estimator)

    # Create a fake (random) true domain parameter
    domain_param_gt = to.tensor([env_real.domain_param[key] for _, key in dp_mapping.items()])
    domain_param_gt += domain_param_gt * to.randn(len(dp_mapping)) / 5
    domain_param_gt = domain_param_gt.unsqueeze(0)
    data_real = simulator(domain_param_gt)

    # Get a (random) condition
    condition = Embedding.pack(domain_param_gt.clone())

    if layout == "inside":
        num_rows, num_cols = len(dp_mapping), len(dp_mapping)
    else:
        num_rows, num_cols = len(dp_mapping) + 1, len(dp_mapping) + 1

    if use_prior:
        grid_bounds = None
    else:
        prior = None
        grid_bounds = to.cat([to.zeros((len(dp_mapping), 1)), to.ones((len(dp_mapping), 1))], dim=1)

    _, axs = plt.subplots(num_rows, num_cols, figsize=(14, 14), tight_layout=True)
    fig = draw_posterior_pairwise_heatmap(
        axs,
        posterior,
        data_real,
        dp_mapping,
        condition,
        prior=prior,
        env_real=env_real,
        marginal_layout=layout,
        grid_bounds=grid_bounds,
        grid_res=100,
        normalize_posterior=False,
        rescale_posterior=True,
        labels=None if labels is None else [""] * len(dp_mapping),
        prob_labels=prob_labels,
    )

    assert fig is not None


@pytest.mark.parametrize("env, policy", [("default_omo", "idle_policy")], indirect=True)
@pytest.mark.parametrize("layout", ["inside", "outside"], ids=["inside", "outside"])
@pytest.mark.parametrize("labels", [None, ["dp_1", "dp_2", "dp_3"]], ids=["no_labels", "labels"])
@pytest.mark.parametrize("legend_labels", [None, ["sim"]], ids=["no_legend", "legend"])
@pytest.mark.parametrize("axis_limits", [None, "use_prior"], ids=["no_limits", "prior_limits"])
@pytest.mark.parametrize("use_trafo", [False, True], ids=["no_trafo", "trafo"])
@pytest.mark.parametrize("use_kde", [False, True], ids=["no_kde", "kde"])
def test_pair_plot_scatter(
    env: SimEnv,
    policy: Policy,
    layout: str,
    labels: Optional[str],
    legend_labels: Optional[str],
    axis_limits: Optional[str],
    use_kde: bool,
    use_trafo: bool,
):
    def _simulator(dp: to.Tensor) -> to.Tensor:
        """The most simple interface of a simulation to sbi, using `env` and `policy` from outer scope"""
        ro = rollout(env, policy, eval=True, reset_kwargs=dict(domain_param=dict(m=dp[0], k=dp[1], d=dp[2])))
        observation_sim = to.from_numpy(ro.observations[-1]).to(dtype=to.float32)
        return to.atleast_2d(observation_sim)

    # Fix the init state
    env.init_space = SingularStateSpace(env.init_space.sample_uniform())
    env_real = deepcopy(env)
    env_real.domain_param = {"m": 0.8, "k": 15, "d": 0.7}

    # Optionally transformed domain parameters for inference
    if use_trafo:
        env = LogDomainParamTransform(env, mask=["k"])

    # Domain parameter mapping and prior
    dp_mapping = {0: "m", 1: "k", 2: "d"}
    k_low = np.log(10) if use_trafo else 10
    k_up = np.log(20) if use_trafo else 20
    prior = BoxUniform(low=to.tensor([0.5, k_low, 0.2]), high=to.tensor([1.5, k_up, 0.8]))

    # Learn a likelihood from the simulator
    density_estimator = posterior_nn(model="maf", hidden_features=10, num_transforms=3)
    snpe = SNPE(prior, density_estimator)
    simulator, prior = prepare_for_sbi(_simulator, prior)
    domain_param, data_sim = simulate_for_sbi(simulator=simulator, proposal=prior, num_simulations=50, num_workers=1)
    snpe.append_simulations(domain_param, data_sim)
    density_estimator = snpe.train(max_num_epochs=5)
    posterior = snpe.build_posterior(density_estimator)

    # Create a fake (random) true domain parameter
    domain_param_gt = to.tensor([env_real.domain_param[dp_mapping[key]] for key in sorted(dp_mapping.keys())])
    domain_param_gt += domain_param_gt * to.randn(len(dp_mapping)) / 10
    domain_param_gt = domain_param_gt.unsqueeze(0)
    data_real = simulator(domain_param_gt)

    domain_params, log_probs = SBIBase.eval_posterior(
        posterior,
        data_real,
        num_samples=6,
        normalize_posterior=False,
        subrtn_sbi_sampling_hparam=dict(sample_with_mcmc=False),
    )
    dp_samples = [domain_params.reshape(1, -1, domain_params.shape[-1]).squeeze()]

    if layout == "inside":
        num_rows, num_cols = len(dp_mapping), len(dp_mapping)
    else:
        num_rows, num_cols = len(dp_mapping) + 1, len(dp_mapping) + 1

    _, axs = plt.subplots(num_rows, num_cols, figsize=(8, 8), tight_layout=True)
    fig = draw_posterior_pairwise_scatter(
        axs=axs,
        dp_samples=dp_samples,
        dp_mapping=dp_mapping,
        prior=prior if axis_limits == "use_prior" else None,
        env_sim=env,
        env_real=env_real,
        axis_limits=axis_limits,
        marginal_layout=layout,
        labels=labels,
        legend_labels=legend_labels,
        use_kde=use_kde,
    )
    assert fig is not None
