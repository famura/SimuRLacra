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
import seaborn as sns
import warnings
import torch as to
from itertools import product
from matplotlib.patches import Patch
from matplotlib import pyplot as plt, patches
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.utils import BoxUniform
from torch.distributions import Distribution
from torch.distributions.uniform import Uniform
from typing import Sequence, Optional, Union, Mapping, Tuple, List, Any

import pyrado
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.sim_base import SimEnv
from pyrado.plotting.utils import draw_sep_cbar
from pyrado.policies.special.mdn import MDN
from pyrado.utils.checks import check_all_types_equal, is_iterable
from pyrado.utils.data_types import merge_dicts
from pyrado.utils.input_output import print_cbt


def draw_distr_evolution(
    ax: plt.Axes,
    distributions: Sequence[Distribution],
    x_grid_limits: Sequence,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    distr_labels: Optional[Sequence[str]] = None,
    grid_res: Optional[int] = 201,
    alpha: Optional[float] = 0.3,
    cmap_name: Optional[str] = "plasma",
    show_legend: Optional[bool] = True,
    title: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the evolution of a sequence of PyTorch probability distributions.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param ax: axis of the figure to plot on
    :param distributions: iterable with the distributions in the order they should be plotted
    :param x_grid_limits: min and max value for the evaluation grid
    :param x_label: label for the x-axis, no label by default
    :param y_label: label for the y-axis, no label by default
    :param distr_labels: label for each of the distributions
    :param grid_res: number of samples for the input (corresponds to x-axis grid_res of the plot)
    :param cmap_name: name of the color map, e.g. 'inferno', 'RdBu', or 'viridis'
    :param alpha: transparency (alpha-value) for the std area
    :param show_legend: flag if the legend entry should be printed, set to True when using multiple subplots
    :param title: title displayed above the (sub)figure, empty string triggers the default title, set to `None` to
                  suppress the title
    :return: handle to the resulting figure
    """
    if not check_all_types_equal(distributions):
        raise pyrado.TypeErr(msg="Types of all distributions have to be identical!")
    if not isinstance(distributions[0], Distribution):
        raise pyrado.TypeErr(msg="Distributions must be PyTorch Distribution instances!")

    if distr_labels is None:
        distr_labels = [rf"iter\_{i}" for i in range(len(distributions))]

    # Get the color map customized to the number of distributions to plot
    cmap = plt.get_cmap(cmap_name)
    ax.set_prop_cycle(color=cmap(np.linspace(0.0, 1.0, max(2, len(distributions)))))

    # Create evaluation grid
    x_gird = to.linspace(x_grid_limits[0], x_grid_limits[1], grid_res)

    # Plot the data
    for i, d in enumerate(distributions):
        probs = to.exp(d.log_prob(x_gird)).detach().cpu().numpy()
        ax.plot(x_gird.numpy(), probs, label=distr_labels[i])
        ax.fill_between(x_gird.detach().cpu().numpy(), np.zeros_like(probs), probs, alpha=alpha)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend(ncol=2)
    if title is not None:
        ax.set_title(title)
    return plt.gcf()


def draw_posterior_distr_1d(
    ax: plt.Axes,
    posterior: Union[DirectPosterior, List[DirectPosterior]],
    observations_real: to.Tensor,
    dp_mapping: Mapping[int, str],
    dim: Union[int, Tuple[int]],
    prior: Optional[BoxUniform] = None,
    env_real: Optional[DomainRandWrapperBuffer] = None,
    condition: Optional[to.Tensor] = None,
    show_prior: bool = False,
    grid_bounds: Optional[Union[to.Tensor, np.ndarray, list]] = None,
    grid_res: Optional[int] = 500,
    normalize_posterior: Optional[bool] = False,
    rescale_posterior: Optional[bool] = False,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    transposed: Optional[bool] = False,
    plot_kwargs: Optional[dict] = None,
) -> plt.Figure:
    r"""
    Evaluate an posterior obtained from the sbi package on a 2-dim grid of domain parameter values.
    Draw every posterior, conditioned on the real-world observation, in a separate plot.

    :param ax: axis of the figure to plot on
    :param posterior: sbi `DirectPosterior` object to evaluate
    :param observations_real: observations from the real-world rollouts a.k.a. $x_o$
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param dim: selected dimension
    :param env_real: real-world environment a.k.a. target domain. Here it is used in case of a sim-2-sim example to
                     infer the ground truth domain parameters
    :param prior: distribution used by sbi as a prior
    :param condition: condition of the posterior, i.e. domain parameters to fix for the non-plotted dimensions
    :param show_prior: display the prior as a box
    :param grid_bounds: explicit bounds for the 2 selected dimensions of the evaluation gird [2 x 2]. Can be set
                        arbitrarily, but should contain the prior if `show_prior` is `True`.
    :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
    :param rescale_posterior: if `True` scale the probabilities to [0, 1], also if `True` the `normalize_posterior`
                              argument is ignored since it would be a wasted computation
    :param grid_res: number of elements on one axis of the evaluation gird
    :param x_label: label for the x-axis, use domain parameter name by default
    :param y_label: label for the y-axis, use domain parameter name by default
    :param plot_kwargs: keyword arguments forwarded to pyplot's `plot()` function for the posterior distribution
    :return: handle to the resulting figure
    """
    if len(dp_mapping) > 1 and condition is None:
        raise pyrado.ValueErr(
            msg="When the posteriors has more than 1 dimension, i.e. there is more than 1 domain "
            "parameter, a condition has to be provided."
        )
    elif len(dp_mapping) > 1 and (condition.numel() != len(dp_mapping)):
        raise pyrado.ShapeErr(given=condition, expected_match=dp_mapping)
    if isinstance(dim, tuple):
        if len(dim) != 1:
            raise pyrado.ShapeErr(given=dim, expected_match=(1,))
        dim = dim[0]
    if not isinstance(grid_res, int):
        raise pyrado.TypeErr(given=grid_res, expected_type=int)

    num_obs_r, dim_obs_r = observations_real.shape

    # Set defaults which can be overwritten by passing plot_kwargs
    plot_kwargs = merge_dicts([dict(), plot_kwargs])

    # Reconstruct ground truth domain parameters if they exist
    if typed_env(env_real, DomainRandWrapperBuffer):
        dp_gt = to.stack([to.stack(list(d.values())) for d in env_real.randomizer.get_params(-1, "list", "torch")])
    elif isinstance(env_real, SimEnv):
        dp_gt = to.tensor([env_real.domain_param[v] for v in dp_mapping.values()])
        dp_gt = to.atleast_2d(dp_gt)
    else:
        dp_gt = None

    # Create the grid
    if grid_bounds is not None:
        grid_bounds = to.as_tensor(grid_bounds, dtype=to.get_default_dtype())
        if not grid_bounds.shape == (1, 2):
            raise pyrado.ShapeErr(given=grid_bounds, expected_match=(1, 2))
    elif isinstance(prior, BoxUniform):
        grid_bounds = to.tensor([[prior.support.lower_bound[dim], prior.support.upper_bound[dim]]])
    else:
        raise NotImplementedError
    grid_x = to.linspace(grid_bounds[0, 0], grid_bounds[0, 1], grid_res)
    if condition is None:
        # No condition is necessary since dim(posterior) = dim(grid) = 1
        grid = to.atleast_2d(grid_x)  # TODO test
    else:
        # A condition is necessary since dim(posterior) > dim(grid) = 1
        grid = condition.repeat(grid_res, 1)
        grid[:, dim] = grid_x
    if not grid.shape == (grid_res, len(dp_mapping)):
        raise pyrado.ShapeErr(given=grid, expected_match=(grid_res, len(dp_mapping)))

    # Compute the posterior probabilities
    if rescale_posterior:
        log_prob = sum([posterior.log_prob(grid, obs, False) for obs in observations_real])
        prob = to.exp(log_prob - log_prob.max())
    else:
        log_prob = sum([posterior.log_prob(grid, obs, normalize_posterior) for obs in observations_real])
        prob = to.exp(log_prob)

    # Plot the posterior
    if transposed:
        ax.plot(prob.numpy(), grid_x.numpy(), **plot_kwargs)
        ax.margins(x=0.03)  # in units of axis span
    else:
        ax.plot(grid_x.numpy(), prob.numpy(), **plot_kwargs)
        ax.margins(y=0.03)  # in units of axis span

    # Plot the ground truth parameters
    if dp_gt is not None:
        if transposed:
            ax.hlines(dp_gt[:, dim], xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors="firebrick")
        else:
            ax.vlines(dp_gt[:, dim], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors="firebrick")

    # Plot bounding box for the prior
    if prior is not None and show_prior:
        raise NotImplementedError  # TODO
        # _draw_prior(ax, prior, dim, dim)

    # Annotate
    if transposed:
        ax.set_ylabel(f"${dp_mapping[dim]}$" if x_label == "" else x_label)
        ax.set_xlabel(rf"log $p({dp_mapping[dim]} | \tau^{{obs}}_{{1:{num_obs_r}}})$" if y_label == "" else y_label)
    else:
        ax.set_xlabel(f"${dp_mapping[dim]}$" if x_label == "" else x_label)
        ax.set_ylabel(rf"log $p({dp_mapping[dim]} | \tau^{{obs}}_{{1:{num_obs_r}}})$" if y_label == "" else y_label)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    ax.set_title(f"")

    return plt.gcf()


def draw_posterior_distr_2d(
    axs: plt.Axes,
    plot_type: str,
    posterior: Union[DirectPosterior, List[DirectPosterior]],
    observations_real: to.Tensor,
    dp_mapping: Mapping[int, str],
    dims: Tuple[int, int],
    prior: Optional[BoxUniform] = None,
    env_real: Optional[DomainRandWrapperBuffer] = None,
    condition: Optional[to.Tensor] = None,
    show_prior: bool = False,
    grid_bounds: Optional[Union[to.Tensor, np.ndarray, list]] = None,
    grid_res: Optional[int] = 500,
    normalize_posterior: Optional[bool] = False,
    rescale_posterior: Optional[bool] = False,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    title: Optional[str] = "",
    add_sep_colorbar: Optional[bool] = False,
    contourf_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
    colorbar_kwargs: Optional[dict] = None,
) -> Union[plt.Figure, Optional[Union[Any, plt.Figure]]]:
    r"""
    Evaluate an posterior obtained from the sbi package on a 2-dim grid of domain parameter values.
    Draw every posterior, conditioned on the real-world observation, in a separate plot.

    :param axs: axis (joint) or axes (separately) of the figure to plot on
    :param plot_type: joint to draw the joint posterior probability probabilities in one plot, or separately to draw
                      the posterior probabilities, conditioned on the real-world observation, in a separate plot. The
                      modes `joint` and `separate` always use the latest posterior (the only one given), while the mode
                      `evolution` uses the posterior from the iteration in which the observation was obtained.
    :param posterior: sbi `DirectPosterior` object to evaluate
    :param observations_real: observations from the real-world rollouts a.k.a. $x_o$
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param prior: distribution used by sbi as a prior
    :param env_real: real-world environment a.k.a. target domain. Here it is used in case of a sim-2-sim example to
                     infer the ground truth domain parameters
    :param dims: selected dimensions
    :param condition: condition of the posterior, i.e. domain parameters to fix for the non-plotted dimensions
    :param show_prior: display the prior as a box
    :param grid_bounds: explicit bounds for the 2 selected dimensions of the evaluation gird [2 x 2]. Can be set
                        arbitrarily, but should contain the prior if `show_prior` is `True`.
    :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
    :param rescale_posterior: if `True` scale the probabilities to [0, 1], also if `True` the `normalize_posterior`
                              argument is ignored since it would be a wasted computation
    :param grid_res: number of elements on one axis of the evaluation gird
    :param x_label: label for the x-axis, use domain parameter name by default
    :param y_label: label for the y-axis, use domain parameter name by default
    :param title: title displayed above the (sub)figure, empty string triggers the default title, set to `None` to
                  suppress the title
    :param add_sep_colorbar: if `True`, add a color bar in a separate figure, else no color bar is plotted
    :param contourf_kwargs: keyword arguments forwarded to pyplot's `contourf()` function for the posterior distribution
    :param scatter_kwargs: keyword arguments forwarded to pyplot's `scatter()` function for the true parameter
    :param colorbar_kwargs: keyword arguments forwarded to `draw_sep_cbar()` function, possible kwargs: `ax_cb`,
                            `colorbar_label`, `colorbar_orientation`, `fig_size`, `cmap`, `norm`, num_major_ticks_cb`
    :return: handle to the resulting figure
    """
    num_obs_r, dim_obs_r = observations_real.shape
    dim_x, dim_y = dims
    plot_type = plot_type.lower()

    # Check the inputs
    if plot_type == "joint":
        if not isinstance(axs, plt.Axes):
            raise pyrado.TypeErr(given=axs, expected_type=plt.Axes)
    elif plot_type in ["separate", "evolution"]:
        axs = np.atleast_2d(axs)
        if not axs.size == num_obs_r:
            raise pyrado.ShapeErr(msg=f"The plotting axes need to be a 2-dim array with {num_obs_r} elements!")
    else:
        raise pyrado.ValueErr(given=plot_type, eq_constraint="joint, separate, or evolution")
    if plot_type in ["joint", "separate"]:
        if not isinstance(posterior, DirectPosterior):
            raise pyrado.TypeErr(given=posterior, expected_type=DirectPosterior)
    elif plot_type == "evolution":
        if not (is_iterable(posterior) and isinstance(posterior[0], DirectPosterior)):
            raise pyrado.TypeErr(given=posterior[0], expected_type=DirectPosterior)
    if len(dp_mapping) == 1:
        raise NotImplementedError("So far, this function does not support plotting 1-dim posteriors.")
    if len(dp_mapping) > 2 and condition is None:
        raise pyrado.ValueErr(
            msg="When the posteriors has more than 2 dimensions, i.e. there are more than 2 domain "
            "parameters, a condition has to be provided."
        )
    elif len(dp_mapping) > 2 and (condition.numel() != len(dp_mapping)):
        raise pyrado.ShapeErr(given=condition, expected_match=dp_mapping)
    if not isinstance(grid_res, int):
        raise pyrado.TypeErr(given=grid_res, expected_type=int)

    # Set defaults which can be overwritten by passing plot_kwargs
    contourf_kwargs = merge_dicts([dict(), contourf_kwargs])
    scatter_kwargs = merge_dicts([dict(zorder=1, s=60, marker="o", c="w", edgecolors="k"), scatter_kwargs])
    colorbar_kwargs = merge_dicts([dict(fig_size=(4, 1)), colorbar_kwargs])

    # Reconstruct ground truth domain parameters if they exist
    if typed_env(env_real, DomainRandWrapperBuffer):
        dp_gt = to.stack([to.stack(list(d.values())) for d in env_real.randomizer.get_params(-1, "list", "torch")])
    elif isinstance(env_real, SimEnv):
        dp_gt = to.tensor([env_real.domain_param[v] for v in dp_mapping.values()])
        dp_gt = to.atleast_2d(dp_gt)
    else:
        dp_gt = None

    # Create the grid
    if grid_bounds is not None:
        grid_bounds = to.as_tensor(grid_bounds, dtype=to.get_default_dtype())
        if not grid_bounds.shape == (2, 2):
            raise pyrado.ShapeErr(given=grid_bounds, expected_match=(2, 2))
    elif isinstance(prior, BoxUniform):
        grid_bounds = to.tensor(
            [
                [prior.support.lower_bound[dim_x], prior.support.upper_bound[dim_x]],
                [prior.support.lower_bound[dim_y], prior.support.upper_bound[dim_y]],
            ]
        )
    else:
        raise NotImplementedError
    x = to.linspace(grid_bounds[0, 0], grid_bounds[0, 1], grid_res)  # 1 2 3
    y = to.linspace(grid_bounds[1, 0], grid_bounds[1, 1], grid_res)  # 4 5 6
    x = x.repeat(grid_res)  # 1 2 3 1 2 3 1 2 3
    y = to.repeat_interleave(y, grid_res)  # 4 4 4 5 5 5 6 6 6
    grid_x, grid_y = x.view(grid_res, grid_res), y.view(grid_res, grid_res)
    grid_x, grid_y = grid_x.numpy(), grid_y.numpy()
    if condition is None:
        # No condition is necessary since dim(posterior) = dim(grid) = 2
        grid = to.stack([x, y], dim=1)
    else:
        # A condition is necessary since dim(posterior) > dim(grid) = 2
        grid = condition.repeat(grid_res ** 2, 1)
        grid[:, dim_x] = x
        grid[:, dim_y] = y
    if not grid.shape == (grid_res ** 2, len(dp_mapping)):
        raise pyrado.ShapeErr(given=grid, expected_match=(grid_res ** 2, len(dp_mapping)))

    fig = plt.gcf()
    if plot_type == "joint":
        # Compute the posterior probabilities
        if rescale_posterior:
            log_prob = sum([posterior.log_prob(grid, obs, False) for obs in observations_real])
            prob = to.exp(log_prob - log_prob.max())
        else:
            log_prob = sum([posterior.log_prob(grid, obs, normalize_posterior) for obs in observations_real])
            prob = to.exp(log_prob)
        prob = prob.reshape(grid_res, grid_res).numpy()

        # Plot the posterior
        axs.contourf(
            grid_x, grid_y, prob, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", **contourf_kwargs
        )

        # Plot the ground truth parameters
        if dp_gt is not None:
            axs.scatter(dp_gt[:, dim_x], dp_gt[:, dim_y], **scatter_kwargs)

        # Plot bounding box for the prior
        if prior is not None and show_prior:
            _draw_prior(axs, prior, dim_x, dim_y)

        # Annotate
        axs.set_aspect(1.0 / axs.get_data_ratio(), adjustable="box")
        axs.set_xlabel(f"${dp_mapping[dim_x]}$" if x_label == "" else x_label)
        axs.set_ylabel(f"${dp_mapping[dim_y]}$" if y_label == "" else y_label)
        axs.set_title(f"{num_obs_r} observations" if title == "" else title)
        fig.canvas.set_window_title("Joint Probability of the Latest Posterior for All Real World Observation")

    elif plot_type in ["separate", "evolution"]:
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                # Compute the posterior probabilities
                idx = j + i * axs.shape[1]  # iterate column-wise
                p = posterior if plot_type == "separate" else posterior[idx]
                log_prob = p.log_prob(grid, observations_real[idx, :], normalize_posterior)
                prob = to.exp(log_prob - log_prob.max())  # scale the probabilities to [0, 1]
                prob = prob.reshape(grid_res, grid_res).numpy()

                # Plot the posterior
                axs[i, j].contourf(
                    grid_x, grid_y, prob, extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", **contourf_kwargs
                )

                # Plot the ground truth parameters
                if dp_gt is not None:
                    axs[i, j].scatter(dp_gt[:, dim_x], dp_gt[:, dim_y], **scatter_kwargs)

                # Plot bounding box for the prior
                if prior is not None and show_prior:
                    _draw_prior(axs[i, j], prior, dim_x, dim_y)

                # Annotate
                axs[i, j].set_aspect(1.0 / axs[i, j].get_data_ratio(), adjustable="box")
                axs[i, j].set_xlabel(f"${dp_mapping[dim_x]}$" if x_label == "" else x_label)
                axs[i, j].set_ylabel(f"${dp_mapping[dim_y]}$" if y_label == "" else y_label)
                axs[i, j].set_title(f"observation {idx}")
        if plot_type == "separate":
            fig.canvas.set_window_title("Probability of the Latest Posterior for Every Real World Observation")
        else:
            fig.canvas.set_window_title("Probability of the Current Posterior for Every Real World Observation")

    # Add a separate colorbar if desired
    fig_cb = draw_sep_cbar(**colorbar_kwargs) if add_sep_colorbar else None

    return fig, fig_cb


def _draw_prior(ax, prior, dim_x, dim_y):
    """ Helper function to draw a rectangle for the prior (assuming uniform distribution) """
    x = prior.support.lower_bound[dim_x]
    y = prior.support.lower_bound[dim_y]
    dx = prior.support.upper_bound[dim_x] - prior.support.lower_bound[dim_x]
    dy = prior.support.upper_bound[dim_y] - prior.support.lower_bound[dim_y]
    rect = patches.Rectangle((x, y), dx, dy, lw=1, ls="--", edgecolor="gray", facecolor="none")
    ax.add_patch(rect)


def draw_posterior_distr_pairwise(
    axs: plt.Axes,
    posterior: Union[DirectPosterior, to.distributions.Distribution],
    observations_real: to.Tensor,
    dp_mapping: Mapping[int, str],
    condition: to.Tensor,
    prior: Optional[Union[BoxUniform, Uniform]] = None,
    env_real: Optional[DomainRandWrapperBuffer] = None,
    grid_res: Optional[int] = 100,
    marginal_layout: str = "classical",
    normalize_posterior: Optional[bool] = False,
    rescale_posterior: Optional[bool] = False,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
) -> plt.Figure:
    """

    :param axs: axis (joint) or axes (separately) of the figure to plot on
    :param posterior: sbi `DirectPosterior` object to evaluate
    :param observations_real: observations from the real-world rollouts a.k.a. $x_o$
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param condition: condition of the posterior, i.e. domain parameters to fix for the non-plotted dimensions
    :param prior: distribution used by sbi as a prior
    :param env_real: real-world environment a.k.a. target domain. Here it is used in case of a sim-2-sim example to
                     infer the ground truth domain parameters
    :param dims: selected dimensions
    :param grid_res: number of elements on one axis of the evaluation gird
    :param marginal_layout: choose between `classical` for plotting the marginals on the diagonal (more dense), and
                           `top-right-posterior` plotting the marginals on the side (better comparison)
    :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
    :param rescale_posterior: if `True` scale the probabilities to [0, 1], also if `True` the `normalize_posterior`
                              argument is ignored since it would be a wasted computation
    :param x_labels: list of labels for the x-axis, use domain parameter name by default
    :param y_labels: list of labels for the y-axis, use domain parameter name by default
    :return: figure containing the pair plot
    """

    if marginal_layout == "classical":
        plot_size = (len(dp_mapping), len(dp_mapping))
    elif marginal_layout == "top-right-posterior":
        plot_size = (len(dp_mapping) + 1, len(dp_mapping) + 1)
    else:
        raise pyrado.ValueErr(given=marginal_layout, eq_constraint="classical or top-right-corner")
    assert isinstance(axs, np.ndarray)
    assert axs.shape == plot_size

    c_palette = sns.color_palette()  # TODO

    # Generate the indices for the subplots
    if marginal_layout == "classical":
        plot_size = (len(dp_mapping), len(dp_mapping))
        # Set the indices for the marginal plots
        idcs_marginal = [[i, i] for i in range(plot_size[0])]
        idcs_skipp = []
        # Set the indices for the pair plots
        idcs_pair = [
            [i, j]
            for i, j in product(list(range(plot_size[0])), list(range(plot_size[1])))
            if [i, j] not in idcs_marginal
        ]

    else:
        plot_size = (len(dp_mapping) + 1, len(dp_mapping) + 1)
        # Set the indices for the marginal plots
        idcs_marginal = [[0, j] for j in range(0, len(dp_mapping))] + [
            [i, len(dp_mapping)] for i in range(1, len(dp_mapping) + 1)
        ]
        idcs_skipp = [[i + 1, i] for i in range(len(dp_mapping))] + [[0, plot_size[1] - 1]]
        # Set the indices for the pair plots
        idcs_pair = [
            [i, j]
            for i, j in product(list(range(1, plot_size[0])), list(range(plot_size[0] - 1)))
            if [i, j] not in idcs_skipp
        ]

    # Reconstruct ground truth domain parameters if they exist
    if typed_env(env_real, DomainRandWrapperBuffer):
        dp_gt = to.stack([to.stack(list(d.values())) for d in env_real.randomizer.get_params(-1, "list", "torch")])
    elif isinstance(env_real, SimEnv):
        # TODO test (not urgent)
        dp_gt = to.tensor([env_real.domain_param[v] for v in dp_mapping.values()])
        dp_gt = to.atleast_2d(dp_gt)
    else:
        dp_gt = None

    # Plot everything
    for i, j in idcs_pair:
        if marginal_layout == "top-right-posterior":
            sam_i = i - 1
            sam_j = j
        else:
            sam_i = i
            sam_j = j
        # Plot the points
        draw_posterior_distr_2d(
            axs[i, j],
            "joint",
            posterior,
            observations_real,
            dp_mapping,
            (sam_j, sam_i),  # TODO @Theo double-check (i think you got it wrong in the first place, check with labels)
            prior,
            env_real,
            condition=condition,
            grid_bounds=None,
            grid_res=grid_res,
            show_prior=False,
            normalize_posterior=normalize_posterior,
            rescale_posterior=rescale_posterior,
            x_label="",
            y_label="",
            title=None,
            add_sep_colorbar=False,
        )

    rotate = None  # initialize invalid
    sam_i = None  # initialize invalid
    for i, j in idcs_marginal:
        if marginal_layout == "top-right-posterior":
            if i == 0:
                sam_i = j
                rotate = False
            if j == len(dp_mapping):
                sam_i = i - 1
                rotate = True
        else:
            sam_i = i
            rotate = False

        # Plot the marginal distribution
        draw_posterior_distr_1d(
            axs[i, j],
            posterior,
            observations_real,
            dp_mapping,
            sam_i,  # TODO @Theo double-check
            prior,
            env_real,
            condition,
            show_prior=False,
            grid_bounds=None,
            grid_res=grid_res,
            x_label="",  # TODO do labeling here
            y_label="",  # TODO do labeling here
            normalize_posterior=normalize_posterior,
            transposed=rotate,
        )

    # Label all axis
    # _labeling_pair_plot(
    #     axs, marginal_layout, dp_mapping, plot_size, idcs_pair, idcs_skipp, idcs_marginal, x_labels, y_labels
    # )
    return plt.gcf()


def draw_pair_plot(
    axs: Optional[plt.Axes],
    dist: Union[DirectPosterior, to.distributions.Distribution],
    dp_mapping: Mapping[int, str],
    condition: to.Tensor,
    observation_real: Optional[to.Tensor] = None,
    prior: Optional[Union[BoxUniform, Uniform]] = None,
    grid_bounds: Optional[Union[to.Tensor, np.ndarray, list]] = None,
    grid_res: Optional[int] = 100,
    num_samples: Optional[int] = 1000,
    reference_samples: to.Tensor = None,
    true_params: to.Tensor = None,
    # density_type: str = "histogram", # TODO: different plotting possibilities for the marginal plots
    marginal_layout: str = "classical",
    normalize_posterior: bool = True,
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
    legend: bool = True,
    use_sns: bool = False,
) -> plt.Figure:
    r"""
    Plotting utility which compares all random variables of a multivariate probability distribution with dim > 2.


    :param axs: axis (joint) or axes (separately) of the figure to plot on
    :param marginal_layout: Choose between 'classical' for plotting the marginals on the diagonal (more dense), and
                      'top-right-posterior' plotting the marginals on the side (better comparison)
    :param dist: distribution from which should be sampled from.
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param condition: condition of the posterior, i.e. domain parameters to fix for the non-plotted dimensions
    :param observation_real: in case dist is a DirectPosterior, the single observation serves as the condition i.e.
                             $p(\theta | tau = observation_{real})$
    :param prior: distribution used by sbi as a prior, should be BoxUniform and hence sets the plot borders
    :param grid_bounds: manually set the plot borders. Either prior or grid_bounds is required.
    :param grid_res: explicit bounds for the 2 selected dimensions of the evaluation gird [2 x 2]. Can be set
                        arbitrarily, but should contain the prior if `show_prior` is `True`.
    :param num_samples: Number of samples used for the
    :param reference_samples: if reference samples are available, plot them as well.
    :param true_params: if true params are available, plot them as well.
    :param normalize_posterior: Choose to normalize the posterior. If False the density plots are scaled by a constant.
    :param x_labels: list of labels for the x-axis, use domain parameter name by default
    :param y_labels: list of labels for the y-axis, use domain parameter name by default
    :param use_sns: use the default pair plot from seaborn. Only for reference and will be removed later
    :param legend: if true a legend will be plotted.
    :return: figure containing the pair plot
    """
    # TODO: check if given axis size is appropriate. Otherwise send message and generate your own size
    if marginal_layout == "classical":
        plot_size = (len(dp_mapping), len(dp_mapping))
    elif marginal_layout == "top-right-posterior":
        plot_size = (len(dp_mapping) + 1, len(dp_mapping) + 1)
    else:
        raise pyrado.ValueErr(given=marginal_layout, eq_constraint="classical or top-right-corner")

    if axs is None:
        print_cbt(f"Generated new plot with {plot_size} shaped sub-plots.", "w")
        tight_layout = False if legend else True
        _, axs = plt.subplots(*plot_size, figsize=(14, 14), tight_layout=tight_layout)

    if marginal_layout == "classical":
        plot_size = (len(dp_mapping), len(dp_mapping))
        # Set the indices for the marginal plots
        idcs_marginal = [[i, i] for i in range(plot_size[0])]
        idcs_skipp = []
        # Set the indices for the pair plots
        idcs_pair = [
            [i, j]
            for i, j in product(list(range(plot_size[0])), list(range(plot_size[1])))
            if [i, j] not in idcs_marginal
        ]

    else:
        plot_size = (len(dp_mapping) + 1, len(dp_mapping) + 1)
        # Set the indices for the marginal plots
        idcs_marginal = [[0, j] for j in range(0, len(dp_mapping))] + [
            [i, len(dp_mapping)] for i in range(1, len(dp_mapping) + 1)
        ]
        idcs_skipp = [[i + 1, i] for i in range(len(dp_mapping))] + [[0, plot_size[1] - 1]]
        # Set the indices for the pair plots
        idcs_pair = [
            [i, j]
            for i, j in product(list(range(1, plot_size[0])), list(range(plot_size[0] - 1)))
            if [i, j] not in idcs_skipp
        ]

    c_palette = sns.color_palette()

    # Set the grid boundaries
    num_params = len(dp_mapping)
    if grid_bounds is not None:
        grid_bounds = to.as_tensor(grid_bounds, dtype=to.get_default_dtype())
        if not grid_bounds.shape == (num_params, 2):
            raise pyrado.ShapeErr(given=grid_bounds, expected_match=(num_params, 2))
    elif isinstance(prior, BoxUniform) or isinstance(prior, Uniform):
        grid_bounds = to.tensor(
            [[prior.support.lower_bound[dim_i], prior.support.upper_bound[dim_i]] for dim_i in range(num_params)]
        )
    else:
        raise NotImplementedError

    log_prob_dict = {}
    if isinstance(dist, DirectPosterior) or isinstance(dist, MDN):
        if observation_real is None:
            raise pyrado.ValueErr(
                given=None,
                msg="DirectPosterior requires an observation for sampling " "and evaluating the log-probability",
            )
        else:
            dist.set_default_x(observation_real)
            log_prob_dict = {"norm_posterior": normalize_posterior}
    elif isinstance(dist, to.distributions.Distribution):
        if observation_real is not None:
            warnings.warn(
                message="Given real observation is not used with the given distribution.\t" "Check if this is intended!"
            )

    #  Draw num_samples samples from observation batch
    if reference_samples is not None:
        perm = to.randperm(reference_samples.shape[0])
        idx = perm[:num_samples]
        reference_samples = reference_samples[idx, :]

    # Sample from the distribution
    samples = dist.sample((num_samples,))

    given_refernce = True if reference_samples is not None else False
    given_true_params = True if true_params is not None else False

    # Plot everything
    if use_sns:
        import pandas as pd

        df = pd.DataFrame(data=samples.numpy())
        sns.pairplot(df)
    else:
        for i, j in idcs_pair:
            if marginal_layout == "top-right-posterior":
                sam_i = i - 1
                sam_j = j
            else:
                sam_i = i
                sam_j = j
            # Plot the points
            plt.sca(axs[i, j])
            sns.scatterplot(x=samples[:, sam_j], y=samples[:, sam_i], color=c_palette[0])
            # axs[i, j].scatter(samples[:, j], samples[:, i], c="blue")
            if reference_samples is not None:
                sns.scatterplot(x=reference_samples[:, sam_j], y=reference_samples[:, sam_i], color=c_palette[1])
                # axs[i, j].scatter(reference_samples[:, j], reference_samples[:, i], c="black")
            if true_params is not None:
                axs[i, j].scatter(true_params[sam_j], true_params[sam_i], color=c_palette[2], marker="x")

        rotate = None  # initialize invalid
        sam_i = None  # initialize invalid
        for i, j in idcs_marginal:
            if marginal_layout == "top-right-posterior":
                if i == 0:
                    sam_i = j
                    rotate = False
                if j == len(dp_mapping):
                    sam_i = i - 1
                    rotate = True
            else:
                sam_i = i
                rotate = False
            plt.sca(axs[i, j])

            # Generate 1-dim grid
            grid_x = to.linspace(grid_bounds[sam_i, 0], grid_bounds[sam_i, 1], grid_res)  # 1 2 3
            grid = condition.repeat(grid_res, 1)
            grid[:, sam_i] = grid_x

            if not grid.shape == (grid_res, len(dp_mapping)):
                raise pyrado.ShapeErr(given=grid, expected_match=(grid_res, len(dp_mapping)))

            # TODO: add possibility to plot the density instead of the histogram
            # calculate log_probs
            # log_prob = dist.log_prob(grid, **log_prob_dict)
            # prob = to.exp(log_prob)  # scale the probabilities to [0, 1]
            # prob = prob.numpy()
            #
            # data = np.stack([grid_x.numpy(), prob])
            # density plot of dimension i
            # axs[i, j].plot(grid_x, prob)
            # sns.kdeplot(data=data)
            # num_bins = 100
            # binwidth = (grid_bounds[i, 1] - grid_bounds[i, 0]) / num_bins
            # bins = np.arange(-grid_bounds[i, 0], grid_bounds[i, 1] + binwidth, binwidth)

            if rotate:
                # Rotate the sub-plot
                sns.histplot(y=samples[:, sam_i], color=c_palette[0])
                if given_refernce:
                    sns.histplot(y=reference_samples[:, sam_i], color=c_palette[1])
                if given_true_params:
                    plt.hlines(
                        true_params[sam_i],
                        xmin=axs[i, j].get_xlim()[0],
                        xmax=axs[i, j].get_xlim()[1],
                        colors=c_palette[2],
                    )
                #     axs[i, j].hlines(y=true_params[sam_i].item(), ymin=0.0, ymax=np.max(prob).item(), colors="green")
            else:
                sns.histplot(x=samples[:, sam_i], color=c_palette[0])
                # sns.distplot(samples[:, sam_i], rug=True, kde=False, color=c_palette[0], vertical=rotate)
                if given_refernce:
                    sns.histplot(reference_samples[:, sam_i], color=c_palette[1])

                # Plot the true parameters if given:
                if given_true_params:
                    plt.vlines(
                        true_params[sam_i],
                        ymin=axs[i, j].get_ylim()[0],
                        ymax=axs[i, j].get_ylim()[1],
                        colors=c_palette[2],
                    )

    # Label all axis
    _labeling_pair_plot(
        axs, marginal_layout, dp_mapping, plot_size, idcs_pair, idcs_skipp, idcs_marginal, x_labels, y_labels
    )
    if legend:
        _legend_pair_plot(axs, colormap=c_palette, is_ref_given=given_refernce, is_gt_given=given_true_params)
    return plt.gcf()


def _labeling_pair_plot(
    axs: plt.Axes,
    plot_type: str,
    dp_mapping: Mapping[int, str],  # TODO use values from dp_mapping as labels, and make the overridable using x_label
    plot_size: Tuple[int, int],
    idcs_pair: List[List[int]],
    idcs_skipp: List[int],
    idcs_marginal: List[List[int]],
    x_labels: Optional[List[str]] = None,
    y_labels: Optional[List[str]] = None,
):
    """ Helper function for labeling the pair plot """
    for i, j in idcs_pair:
        if plot_type == "top-right-posterior":
            sam_i = i - 1
        else:
            sam_i = i
        if j == 0:
            axs[i, j].set_ylabel(rf"$\xi_{{{sam_i + 1}}}$" if y_labels is None else y_labels[sam_i], rotation=0)
        if i == plot_size[1] - 1:
            axs[i, j].set_xlabel(rf"$\xi_{{{j + 1}}}$" if x_labels is None else x_labels[j], rotation=0)
        if i != plot_size[1] - 1:
            axs[i, j].set_xticklabels([])
            axs[i, j].set_xlabel("")
        if j != 0:
            axs[i, j].set_yticklabels([])
            axs[i, j].set_ylabel("")

        # axs[i, j].set(xlim=(grid_bounds[j, 0], grid_bounds[j, 1]), ylim=(grid_bounds[i, 0], grid_bounds[i, 1]))

    for i, j in idcs_skipp:
        axs[i, j].set(frame_on=False)
        axs[i, j].set_xticklabels([])
        axs[i, j].set_yticklabels([])
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        if j == 0:
            axs[i, j].set_ylabel(rf"$\xi_{{{i}}}$" if y_labels is None else y_labels[i - 1], rotation=0)
        elif i == plot_size[0] - 1:
            axs[i, j].set_xlabel(rf"$\xi_{{{j + 1}}}$" if x_labels is None else x_labels[j], rotation=0)

    for i, j in idcs_marginal:
        if i != plot_size[1] - 1:
            axs[i, j].set_xticklabels([])
            axs[i, j].set_xlabel("")
        if j != 0:
            axs[i, j].set_yticklabels([])
            axs[i, j].set_ylabel("")

        if plot_type != "top-right-posterior":
            if j == 0:
                axs[i, j].set_ylabel(rf"$\xi_{{{i + 1}}}$" if y_labels is None else y_labels[i], rotation=0)
            if i == plot_size[1] - 1:
                axs[i, j].set_xlabel(rf"$\xi_{{{j + 1}}}$" if x_labels is None else x_labels[j], rotation=0)
        else:
            if i == 0 and j == 0:
                axs[i, j].set_ylabel(rf"$p(\xi_{{i}} | \tau^{{obs}})$")
                axs[i, j].set_yticklabels([])
            elif i == plot_size[0] - 1 and j == plot_size[1] - 1:
                axs[i, j].set_xlabel(rf"$p(\xi_{{i}} | \tau^{{obs}})$", rotation=0)
                axs[i, j].set_xticklabels([])

    plt.gcf().align_xlabels(axs[plot_size[0] - 1, :])
    plt.gcf().align_ylabels(axs[:, 0])


def _legend_pair_plot(
    axs: plt.Axes,
    colormap,
    is_ref_given: Optional[bool] = False,
    is_gt_given: Optional[bool] = False,
):
    """Helper function to plot the legend. This solution is copied from seaborn pairplot"""
    # Create handles
    handles = [
        # Line2D([0], [0], color=colormap[0], lw=4, label='approx. posterior'),
        # Line2D([0], [0], markerfacecolor=colormap[0], color='w', marker='o', label='approx. posterior')
        Patch(facecolor=colormap[0], edgecolor="w", label="approx. posterior")
    ]
    if is_ref_given:
        handles.append(Patch(facecolor=colormap[1], edgecolor="w", label="ref. samples"))
    if is_gt_given:
        handles.append(Patch(facecolor=colormap[2], edgecolor="w", label="ground truth"))

    fig = plt.gcf()
    figlegend = fig.legend(handles=handles, loc="center right", frameon=False)

    # Draw the plot to set the bounding boxes correctly
    _draw_figure(fig)

    # Calculate and set the new width of the figure so the legend fits
    legend_width = figlegend.get_window_extent().width / fig.dpi
    fig_width, fig_height = fig.get_size_inches()
    fig.set_size_inches(fig_width + legend_width, fig_height)

    # Draw the plot again to get the new transformations
    _draw_figure(fig)

    # # Now calculate how much space we need on the right side
    legend_width = figlegend.get_window_extent().width / fig.dpi
    space_needed = legend_width / (fig_width + legend_width)

    margin = 0.01
    # # margin = .04 if self._margin_titles else .01
    space_needed = margin + space_needed
    right = 1 - space_needed

    # Place the subplot axes to give space for the legend
    fig.subplots_adjust(right=right)
    fig.tight_layout(rect=[0, 0, right, 1])


def _draw_figure(fig):
    """Force draw of a matplotlib figure, accounting for back-compat. Also drawn from seaborn pairplot"""
    # See https://github.com/matplotlib/matplotlib/issues/19197 for context
    fig.canvas.draw()
    if fig.stale:
        try:
            fig.draw(fig.canvas.get_renderer())
        except AttributeError:
            pass
