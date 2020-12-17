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

from matplotlib import pyplot as plt
from pyrado.plotting.categorical import draw_categorical
from pyrado.plotting.curve import draw_curve_from_data, draw_dts
from pyrado.plotting.rollout_based import (
    draw_observations_actions_rewards,
    draw_observations,
    draw_actions,
    draw_rewards,
    draw_potentials,
    draw_features,
)
from pyrado.plotting.surface import draw_surface
from pyrado.policies.feed_forward.linear import LinearPolicy
from pyrado.policies.recurrent.potential_based import PotentialBasedPolicy
from pyrado.sampling.rollout import rollout
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
        draw_features(ro, policy)
    elif isinstance(policy, PotentialBasedPolicy):
        draw_potentials(ro)
    else:
        draw_observations_actions_rewards(ro)
        draw_observations(ro)
        draw_actions(ro, env)
        draw_rewards(ro)
        draw_dts(ro.dts_policy, ro.dts_step, ro.dts_remainder, y_top_lim=5)
