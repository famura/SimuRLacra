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
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import seaborn as sns
import torch as to
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter
from sbi.inference.posteriors.direct_posterior import DirectPosterior
from sbi.utils import BoxUniform
from torch.distributions import Distribution, MultivariateNormal
from torch.distributions.uniform import Uniform

import pyrado
from pyrado.environment_wrappers.domain_randomization import DomainRandWrapperBuffer
from pyrado.environment_wrappers.utils import typed_env
from pyrado.environments.sim_base import SimEnv
from pyrado.plotting.utils import draw_sep_cbar
from pyrado.utils.checks import check_all_lengths_equal, check_all_types_equal, is_iterable
from pyrado.utils.data_types import merge_dicts
from pyrado.utils.input_output import completion_context


@to.no_grad()
def draw_distr_evolution(
    ax: plt.Axes,
    distributions: Sequence[Distribution],
    x_grid_limits: Sequence,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    distr_labels: Optional[Sequence[str]] = None,
    grid_res: Optional[int] = 201,
    alpha: float = 0.3,
    cmap_name: Optional[str] = "plasma",
    show_legend: bool = True,
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


@to.no_grad()
def draw_posterior_distr_1d(
    ax: plt.Axes,
    posterior: Union[DirectPosterior, List[DirectPosterior]],
    data_real: to.Tensor,
    dp_mapping: Mapping[int, str],
    dim: Union[int, Tuple[int]],
    prior: Optional[BoxUniform] = None,
    env_real: Optional[DomainRandWrapperBuffer] = None,
    prob: Optional[to.Tensor] = None,
    condition: Optional[to.Tensor] = None,
    show_prior: bool = False,
    grid_bounds: Optional[Union[to.Tensor, np.ndarray, list]] = None,
    grid_res: int = 500,
    normalize_posterior: bool = False,
    rescale_posterior: bool = False,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    transposed: bool = False,
    plot_kwargs: Optional[dict] = None,
) -> plt.Figure:
    r"""
    Evaluate an posterior obtained from the sbi package on a 2-dim grid of domain parameter values.
    Draw every posterior, conditioned on the real-world data, in a separate plot.

    :param ax: axis of the figure to plot on
    :param posterior: sbi `DirectPosterior` object to evaluate
    :param data_real: data from the real-world rollouts a.k.a. set of $x_o$ of shape
                      [num_iter, num_rollouts_per_iter, time_series_length, dim_data]
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param dim: selected dimension
    :param env_real: real-world environment a.k.a. target domain. Here it is used in case of a sim-2-sim example to
                     infer the ground truth domain parameters
    :param prior: distribution used by sbi as a prior
    :param prob: pre-computed probabilities used to compute the marginal probabilities. The use case in mind is the
                 pairwise density plot, where we evaluate several 2-dim grids and later want to use these
                 evaluations to plot a marginal which is in-line with the other plots
    :param condition: condition of the posterior, i.e. domain parameters to fix for the non-plotted dimensions. `None`
                      can be used in case of a 1-dim domain parameter mapping, else it must be a tensor of shape
                      [num_iter, 1, dim_domain_param]
    :param show_prior: display the prior as a box
    :param grid_bounds: explicit bounds for the 2 selected dimensions of the evaluation gird [2 x 2]. Can be set
                        arbitrarily, but should contain the prior if `show_prior` is `True`.
    :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
    :param rescale_posterior: if `True` scale the probabilities to [0, 1], also if `True` the `normalize_posterior`
                              argument is ignored since it would be a wasted computation
    :param grid_res: number of elements on one axis of the evaluation gird
    :param x_label: label for the x-axis, use domain parameter name by default
    :param y_label: label for the y-axis, use domain parameter name by default
    :param transposed: if `True`, plot the x and y axes
    :param plot_kwargs: keyword arguments forwarded to pyplot's `plot()` function for the posterior distribution
    :return: handle to the resulting figure
    """
    if not data_real.ndim == 2:
        raise pyrado.ShapeErr(
            msg=f"The target domain data tensor to have 2 dimensions, but is of shape {data_real.shape}!"
        )
    num_iter = data_real.shape[0]

    # Check the inputs
    if isinstance(dim, tuple):
        if len(dim) != 1:
            raise pyrado.ShapeErr(given=dim, expected_match=(1,))
        dim = dim[0]
    if not isinstance(grid_res, int):
        raise pyrado.TypeErr(given=grid_res, expected_type=int)
    if condition is None:
        # No condition was given, check if that is feasible
        if len(dp_mapping) > 1:
            raise pyrado.ValueErr(
                msg="When the posteriors has more than 1 dimensions, i.e. there are more than 1 domain "
                "parameters, a condition has to be provided."
            )
    else:
        # A condition was given, check it
        if condition.shape != (num_iter, 1, len(dp_mapping)):
            raise pyrado.ShapeErr(given=condition, expected_match=(num_iter, 1, len(dp_mapping)))

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
        if not hasattr(prior, "base_dist"):
            raise AttributeError(
                "The prior does not have the attribute base_distr! Maybe you are using a sbi version < 0.15."
            )
        grid_bounds = to.tensor([[prior.base_dist.support.lower_bound[dim], prior.base_dist.support.upper_bound[dim]]])
    else:
        raise pyrado.ValueErr(msg="Neither an explicit grid nor a prior has been provided!")
    grid_x = to.linspace(grid_bounds[0, 0], grid_bounds[0, 1], grid_res)
    if prob is not None and not isinstance(prob, to.Tensor):
        raise pyrado.TypeErr(given=prob, expected_type=to.Tensor)
    if condition is None:
        # No condition is necessary since dim(posterior) = dim(grid) = 1
        grids = grid_x.view(-1, 1).repeat(num_iter, 1, 1)
    else:
        # A condition is necessary since dim(posterior) > dim(grid) = 1
        grids = condition.repeat(1, grid_res, 1)
        grids[:, :, dim] = grid_x
    if grids.shape != (num_iter, grid_res, len(dp_mapping)):
        raise pyrado.ShapeErr(given=grids, expected_match=(num_iter, grid_res, len(dp_mapping)))

    if prob is None:
        # Compute the posterior probabilities
        if rescale_posterior:
            log_prob = sum([posterior.log_prob(grid, obs, False) for grid, obs in zip(grids, data_real)])
            prob = to.exp(log_prob - log_prob.max())  # scale the probabilities to [0, 1]
        else:
            log_prob = sum([posterior.log_prob(grid, obs, normalize_posterior) for grid, obs in zip(grids, data_real)])
            prob = to.exp(log_prob)
    else:
        # Use precomputed posterior probabilities
        if prob.shape != (grid_res,):
            raise pyrado.ShapeErr(given=prob, expected_match=(grid_res,))

    # Plot the posterior
    if transposed:
        ax.plot(prob.numpy(), grid_x.numpy(), **plot_kwargs)
        ax.margins(x=0.03, y=0)  # in units of axis span
    else:
        ax.plot(grid_x.numpy(), prob.numpy(), **plot_kwargs)
        ax.margins(x=0, y=0.03)  # in units of axis span

    # Plot the ground truth parameters
    if dp_gt is not None:
        if transposed:
            ax.hlines(dp_gt[:, dim], xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors="k")  # firebrick
        else:
            ax.vlines(dp_gt[:, dim], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors="k")  # firebrick

    # Plot bounding box for the prior
    if prior is not None and show_prior:
        _draw_prior(ax, prior, dim, None, num_dim=1, transposed=transposed)

    # Annotate
    if transposed:
        ax.set_ylabel(f"${dp_mapping[dim]}$" if x_label == "" else x_label)
        ax.set_xlabel(rf"log $p({dp_mapping[dim]} | \tau^{{obs}}_{{1:{num_iter}}})$" if y_label == "" else y_label)
    else:
        ax.set_xlabel(f"${dp_mapping[dim]}$" if x_label == "" else x_label)
        ax.set_ylabel(rf"log $p({dp_mapping[dim]} | \tau^{{obs}}_{{1:{num_iter}}})$" if y_label == "" else y_label)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")
    ax.set_title(f"")

    return plt.gcf()


@to.no_grad()
def draw_posterior_distr_2d(
    axs: plt.Axes,
    plot_type: str,
    posterior: Union[DirectPosterior, List[DirectPosterior]],
    data_real: to.Tensor,
    dp_mapping: Mapping[int, str],
    dims: Tuple[int, int],
    prior: Optional[BoxUniform] = None,
    env_real: Optional[DomainRandWrapperBuffer] = None,
    condition: Optional[to.Tensor] = None,
    show_prior: bool = False,
    grid_bounds: Optional[Union[to.Tensor, np.ndarray, list]] = None,
    grid_res: int = 100,
    normalize_posterior: bool = False,
    rescale_posterior: bool = False,
    x_label: Optional[str] = "",
    y_label: Optional[str] = "",
    title: Optional[str] = "",
    add_sep_colorbar: bool = False,
    contourf_kwargs: Optional[dict] = None,
    scatter_kwargs: Optional[dict] = None,
    colorbar_kwargs: Optional[dict] = None,
) -> Union[plt.Figure, Optional[Union[Any, plt.Figure]], to.Tensor]:
    r"""
    Evaluate an posterior obtained from the sbi package on a 2-dim grid of domain parameter values.
    Draw every posterior, conditioned on the real-world data, in a separate plot.

    :param axs: axis (joint) or axes (separately) of the figure to plot on
    :param plot_type: joint to draw the joint posterior probability probabilities in one plot, or separately to draw
                      the posterior probabilities, conditioned on the real-world data, in a separate plot. The
                      modes `joint` and `separate` always use the latest posterior (the only one given), while the mode
                      `evolution` uses the posterior from the iteration in which the data was obtained.
    :param posterior: sbi `DirectPosterior` object to evaluate
    :param data_real: data from the real-world rollouts a.k.a. set of $x_o$ of shape
                      [num_iter, num_rollouts_per_iter, time_series_length, dim_data]
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param prior: distribution used by sbi as a prior
    :param env_real: real-world environment a.k.a. target domain. Here it is used in case of a sim-2-sim example to
                     infer the ground truth domain parameters
    :param dims: selected dimensions
    :param condition: condition of the posterior, i.e. domain parameters to fix for the non-plotted dimensions. `None`
                      can be used in case of a 2-dim domain parameter mapping, else it must be a tensor of shape
                      [num_iter, 1, dim_domain_param]
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
    :return: handle to the resulting figure, optionally the handle to a color bar, and the tensor of the marginal
             probabilities obtained averaging over the rows and columns of the the 2-dim evaluation grid
    """
    # Check the inputs
    if not data_real.ndim == 2:
        raise pyrado.ShapeErr(
            msg=f"The target domain data tensor to have 2 dimensions, but is of shape {data_real.shape}!"
        )
    num_iter = data_real.shape[0]
    dim_x, dim_y = dims

    plot_type = plot_type.lower()
    if plot_type == "joint":
        if not isinstance(axs, plt.Axes):
            raise pyrado.TypeErr(given=axs, expected_type=plt.Axes)
    elif plot_type in ["separate", "evolution-iter", "evolution-round"]:
        axs = np.atleast_2d(axs)
        if not axs.size == num_iter:
            raise pyrado.ShapeErr(msg=f"The plotting axes need to be a 2-dim array with {num_iter} elements!")
    else:
        raise pyrado.ValueErr(given=plot_type, eq_constraint="joint, separate, evolution-iter, or evolution-round")
    if plot_type in ["joint", "separate"]:
        if not isinstance(posterior, DirectPosterior):
            raise pyrado.TypeErr(given=posterior, expected_type=DirectPosterior)
    elif "evolution" in plot_type:
        if not (is_iterable(posterior) and isinstance(posterior[0], DirectPosterior)):
            raise pyrado.TypeErr(given=posterior[0], expected_type=DirectPosterior)
    if not isinstance(grid_res, int):
        raise pyrado.TypeErr(given=grid_res, expected_type=int)
    if len(dp_mapping) == 1:
        raise NotImplementedError("The draw_posterior_distr_2d() function does not support plotting 1-dim posteriors.")
    if condition is None:
        # No condition was given, check if that is feasible
        if len(dp_mapping) > 2:
            raise pyrado.ValueErr(
                msg="When the posteriors has more than 2 dimensions, i.e. there are more than 2 domain "
                "parameters, a condition has to be provided."
            )
    else:
        # A condition was given, check it
        if condition.shape != (num_iter, 1, len(dp_mapping)):
            raise pyrado.ShapeErr(given=condition, expected_match=(num_iter, 1, len(dp_mapping)))

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
        if grid_bounds.shape != (2, 2):
            raise pyrado.ShapeErr(given=grid_bounds, expected_match=(2, 2))
    elif isinstance(prior, BoxUniform):
        if not hasattr(prior, "base_dist"):
            raise AttributeError(
                "The prior does not have the attribute base_distr! Maybe you are using a sbi version < 0.15."
            )
        grid_bounds = to.tensor(
            [
                [prior.base_dist.support.lower_bound[dim_x], prior.base_dist.support.upper_bound[dim_x]],
                [prior.base_dist.support.lower_bound[dim_y], prior.base_dist.support.upper_bound[dim_y]],
            ]
        )
    elif isinstance(prior, MultivariateNormal):
        # Construct a grid with +/-3 prior std around the prior mean
        lb = prior.mean - 3 * to.sqrt(prior.variance)
        ub = prior.mean + 3 * to.sqrt(prior.variance)
        grid_bounds = to.tensor([[lb[dim_x], ub[dim_x]], [lb[dim_y], ub[dim_y]]])
    else:
        raise pyrado.ValueErr(msg="Neither an explicit grid nor a prior has been provided!")
    x = to.linspace(grid_bounds[0, 0].item(), grid_bounds[0, 1].item(), grid_res)  # 1 2 3
    y = to.linspace(grid_bounds[1, 0].item(), grid_bounds[1, 1].item(), grid_res)  # 4 5 6
    x = x.repeat(grid_res)  # 1 2 3 1 2 3 1 2 3
    y = to.repeat_interleave(y, grid_res)  # 4 4 4 5 5 5 6 6 6
    grid_x, grid_y = x.view(grid_res, grid_res), y.view(grid_res, grid_res)
    if condition is None:
        # No condition is necessary since dim(posterior) = dim(grid) = 2
        grids = to.stack([x, y], dim=1).repeat(num_iter, 1, 1)
    else:
        # A condition is necessary since dim(posterior) > dim(grid) = 2
        grids = condition.repeat(1, grid_res ** 2, 1)
        grids[:, :, dim_x] = x
        grids[:, :, dim_y] = y
    if grids.shape != (num_iter, grid_res ** 2, len(dp_mapping)):
        raise pyrado.ShapeErr(given=grids, expected_match=(grid_res ** 2, len(dp_mapping)))

    fig = plt.gcf()
    if plot_type == "joint":
        # Compute the posterior probabilities
        with completion_context("Evaluating domain param grid", color="w"):
            if rescale_posterior:
                log_prob = sum([posterior.log_prob(grid, obs, False) for grid, obs in zip(grids, data_real)])
                prob = to.exp(log_prob - log_prob.max())  # scale the probabilities to [0, 1]
            else:
                log_prob = sum(
                    [posterior.log_prob(grid, obs, normalize_posterior) for grid, obs in zip(grids, data_real)]
                )
                prob = to.exp(log_prob)
        prob = prob.reshape(grid_res, grid_res)

        # Plot the posterior
        axs.contourf(
            grid_x.numpy(),
            grid_y.numpy(),
            prob.numpy(),
            extent=[x.min(), x.max(), y.min(), y.max()],
            origin="lower",
            **contourf_kwargs,
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
        axs.set_title(f"across {num_iter} iterations" if title == "" else title)
        fig.canvas.set_window_title("Posterior Probability")

    elif plot_type in ["separate", "evolution-iter", "evolution-round"]:
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                # Compute the posterior probabilities
                idx = j + i * axs.shape[1]  # iterate column-wise
                p = posterior if plot_type == "separate" else posterior[idx]
                with completion_context("Evaluating domain param grid", color="w"):
                    if rescale_posterior:
                        log_prob = p.log_prob(grids[idx], data_real[idx], False)
                        prob = to.exp(log_prob - log_prob.max())  # scale the probabilities to [0, 1]
                    else:
                        log_prob = p.log_prob(grids[idx], data_real[idx], normalize_posterior)
                        prob = to.exp(log_prob)
                prob = prob.reshape(grid_res, grid_res)

                # Plot the posterior
                axs[i, j].contourf(
                    grid_x.numpy(),
                    grid_y.numpy(),
                    prob.numpy(),
                    extent=[x.min(), x.max(), y.min(), y.max()],
                    origin="lower",
                    **contourf_kwargs,
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
                default_title = f"iteration {idx}" if plot_type != "evolution-round" else f"round {idx}"
                axs[i, j].set_title(default_title if title == "" else title)

        if plot_type == "separate":
            fig.canvas.set_window_title("Probability of the Latest Posterior")
        else:
            fig.canvas.set_window_title("Probability of the Associated Iteration's Posterior")

    # Add a separate colorbar if desired
    fig_cb = draw_sep_cbar(**colorbar_kwargs) if add_sep_colorbar else None

    # Marginalize
    marginal_prob = to.zeros((2, grid_res))
    marginal_prob[0, :] = to.mean(prob, dim=0)
    marginal_prob[1, :] = to.mean(prob, dim=1)

    return fig, fig_cb, marginal_prob


def _draw_prior(ax, prior: BoxUniform, dim_x: int, dim_y: int, num_dim: int = 2, transposed: bool = False):
    """Helper function to draw a rectangle for the prior (assuming uniform distribution)"""
    if not hasattr(prior, "base_dist"):
        raise AttributeError(
            "The prior does not have the attribute base_distr! Maybe you are using a sbi version < 0.15."
        )

    if num_dim == 1:
        if transposed:
            y = [
                prior.base_dist.support.lower_bound[dim_x],
                prior.base_dist.support.upper_bound[dim_x],
            ]  # double-use of x
            ax.hlines(y, 0, 1, transform=ax.get_xaxis_transform(), lw=1, ls="--", edgecolor="gray", facecolor="none")
        else:
            x = [prior.base_dist.support.lower_bound[dim_x], prior.base_dist.support.upper_bound[dim_x]]
            ax.vlines(x, 0, 1, transform=ax.get_xaxis_transform(), lw=1, ls="--", edgecolor="gray", facecolor="none")

    elif num_dim == 2:
        x = prior.base_dist.support.lower_bound[dim_x]
        y = prior.base_dist.support.lower_bound[dim_y]
        dx = prior.base_dist.support.upper_bound[dim_x] - prior.base_dist.support.lower_bound[dim_x]
        dy = prior.base_dist.support.upper_bound[dim_y] - prior.base_dist.support.lower_bound[dim_y]
        rect = patches.Rectangle((x, y), dx, dy, lw=1, ls="--", edgecolor="gray", facecolor="none")
        ax.add_patch(rect)

    else:
        return NotImplementedError


@to.no_grad()
def draw_posterior_distr_pairwise_heatmap(
    axs: plt.Axes,
    posterior: Union[DirectPosterior, to.distributions.Distribution],
    data_real: to.Tensor,
    dp_mapping: Mapping[int, str],
    condition: to.Tensor,
    prior: Optional[Union[BoxUniform, Uniform]] = None,
    env_real: Optional[DomainRandWrapperBuffer] = None,
    show_prior: bool = False,
    grid_res: int = 100,
    marginal_layout: str = "inside",
    normalize_posterior: bool = False,
    rescale_posterior: bool = False,
    x_labels: Optional[np.ndarray] = "",
    y_labels: Optional[np.ndarray] = "",
    prob_labels: Optional[np.ndarray] = "",
) -> plt.Figure:
    """
    Plot a 2-dim gird of pairwise slices of the posterior distribution evaluated on a grid across these two dimensions,
    while the other dimensions are fixed to the value that is provided as condition.

    :param axs: axis (joint) or axes (separately) of the figure to plot on
    :param posterior: sbi `DirectPosterior` object to evaluate
    :param data_real: data from the real-world rollouts a.k.a. set of $x_o$ of shape
                      [num_iter, num_rollouts_per_iter, time_series_length, dim_data]
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param condition: condition of the posterior, i.e. domain parameters to fix for the non-plotted dimensions. `None`
                      can be used in case of a 2-dim domain parameter mapping, else it must be a tensor of shape
                      [num_iter, 1, dim_domain_param]
    :param prior: distribution used by sbi as a prior
    :param env_real: real-world environment a.k.a. target domain. Here it is used in case of a sim-2-sim example to
                     infer the ground truth domain parameters
    :param show_prior: display the prior as a box or as two lines
    :param dims: selected dimensions
    :param grid_res: number of elements on one axis of the evaluation gird
    :param marginal_layout: choose between `inside` for plotting the marginals on the diagonal (more dense), and
                           `outside` plotting the marginals on the side (better comparison)
    :param normalize_posterior: if `True` the normalization of the posterior density is enforced by sbi
    :param rescale_posterior: if `True` scale the probabilities to [0, 1], also if `True` the `normalize_posterior`
                              argument is ignored since it would be a wasted computation
    :param x_labels: 2-dim numpy array of labels for the x-axes, pass `""` to use domain parameter name by default,
                     or pass `None` to use no labels
    :param y_labels: 2-dim numpy array of labels for the y-axes, pass `""` to use domain parameter name by default,
                     or pass `None` to use no labels
    :param prob_labels: 1-dim numpy array of labels for the probability axis in the marginal plots, pass `""` to use
                        the default labels or pass `None` to use no labels
    :return: figure containing the pair plot
    """
    # Check the inputs
    if marginal_layout == "inside":
        plot_shape = (len(dp_mapping), len(dp_mapping))
    elif marginal_layout == "outside":
        plot_shape = (len(dp_mapping) + 1, len(dp_mapping) + 1)
    else:
        raise pyrado.ValueErr(given=marginal_layout, eq_constraint="inside or outside")
    assert isinstance(axs, np.ndarray)
    if axs.shape != plot_shape:
        raise pyrado.ShapeErr(given=axs, expected_match=plot_shape)

    # Manage the labels
    if x_labels == "":
        # The default values for the labels has been given, fill with the domain parameter names
        x_labels = np.empty(plot_shape, dtype=object)
        for i in range(len(dp_mapping)):
            x_labels[i, :] = dp_mapping[i]
    elif x_labels is not None:
        # A non-default values for the labels has been given, check the shape
        if x_labels.shape != plot_shape:
            raise pyrado.ShapeErr(given=x_labels, expected_match=plot_shape)
    if y_labels == "":
        # The default values for the labels has been given, fill with the domain parameter names
        y_labels = np.empty(plot_shape, dtype=object)
        for i in range(len(dp_mapping)):
            y_labels[:, i] = dp_mapping[i]
    elif y_labels != "" and y_labels is not None:
        # A non-default values for the labels has been given, check the shape
        if y_labels.shape != plot_shape:
            raise pyrado.ShapeErr(given=y_labels, expected_match=plot_shape)
    if marginal_layout == "inside":
        # Remove the repetitive labels, mind the later transposing for the dim assignment
        if x_labels is not None:
            x_labels.T[:-1, :] = None
        if y_labels is not None:
            y_labels.T[:, 1:] = None
    elif marginal_layout == "outside":
        # Remove the repetitive labels, mind the later transposing and shift for the dim assignment
        if x_labels is not None:
            x_labels.T[:-2, :] = None
        if y_labels is not None:
            y_labels.T[:, 1:] = None

    if prob_labels == "":
        prob_labels = [f"p({v})" for v in dp_mapping.values()]
    elif prob_labels != "" and prob_labels is not None:
        if len(prob_labels) != len(dp_mapping):
            raise pyrado.ShapeErr(given=prob_labels, expected_match=dp_mapping)

    # Generate the indices for the subplots
    if marginal_layout == "inside":
        # Set the indices for the marginal plots
        idcs_marginal = [idx for idx, _ in np.ndenumerate(axs) if idx[0] == idx[1]]
        # Set the indices for the pair plots
        idcs_pair = [idx for idx, _ in np.ndenumerate(axs) if idx not in idcs_marginal]
    elif marginal_layout == "outside":
        # Ignore the -1 lower diagonal and the single top right one
        idcs_skipp = [(i + 1, i) for i in range(plot_shape[0] - 1)] + [(0, plot_shape[1] - 1)]
        # Set the indices for the marginal plots
        idcs_marginal = [
            idx
            for idx, _ in np.ndenumerate(axs)
            if (idx[0] == 0 or idx[1] == plot_shape[1] - 1) and (idx not in idcs_skipp)
        ]
        # Set the indices for the pair plots
        idcs_pair = [idx for idx, _ in np.ndenumerate(axs) if idx not in idcs_marginal and idx not in idcs_skipp]

    assert check_all_types_equal(idcs_marginal) and check_all_lengths_equal(idcs_marginal)
    assert check_all_types_equal(idcs_pair) and check_all_lengths_equal(idcs_pair)

    # Initialize a container for the probabilities of each 2-dim grid evaluation
    marginal_probs = to.zeros((len(dp_mapping), grid_res))

    # Plot the pairwise posteriors
    for i, j in idcs_pair:
        dim_x, dim_y = j, i  # transposed since we want the x-axis of the domain params to be the same for all columns
        if marginal_layout == "outside":
            dim_y = i - 1  # counter the shift that we got from the marginal plots in the top row

        _, _, marginal_prob = draw_posterior_distr_2d(
            axs[i, j],
            "joint",
            posterior,
            data_real,
            dp_mapping,
            dims=(dim_x, dim_y),
            prior=prior,
            env_real=env_real,
            condition=condition,
            grid_bounds=None,
            grid_res=grid_res,
            show_prior=show_prior,
            normalize_posterior=normalize_posterior,
            rescale_posterior=rescale_posterior,
            x_label=x_labels[dim_x, dim_y] if x_labels is not None else None,
            y_label=y_labels[dim_x, dim_y] if y_labels is not None else None,
            title=None,
            add_sep_colorbar=False,
        )

        # Extract the marginals (1st dim is always the x-axis in the 2-dim plots)
        marginal_probs[dim_x, :] += marginal_prob[0]
        marginal_probs[dim_y, :] += marginal_prob[1]

    # Every domain parameter got evaluated for every other domain parameter twice, thus rescale
    marginal_probs /= 2 * (len(dp_mapping) - 1)

    # Plot the marginal distributions
    for i, j in idcs_marginal:
        dim = j
        rotate = False
        if marginal_layout == "inside":
            x_label = x_labels[i, j] if x_labels is not None else None
            y_label = prob_labels[i] if prob_labels is not None else None
        elif marginal_layout == "outside":
            if i == 0:
                dim = j
                x_label = x_labels[dim, 0] if x_labels is not None else None
                y_label = prob_labels[dim] if prob_labels is not None else None
            elif j == len(dp_mapping):
                dim = i - 1
                rotate = True
                x_label = x_labels[dim, 0] if x_labels is not None else None
                y_label = prob_labels[dim] if prob_labels is not None else None
        else:
            x_label = y_label = None

        draw_posterior_distr_1d(
            axs[i, j],
            posterior,
            data_real,
            dp_mapping,
            dim,
            prior=prior,
            env_real=env_real,
            prob=marginal_probs[dim],
            condition=condition,
            show_prior=show_prior,
            grid_bounds=None,
            grid_res=grid_res,
            normalize_posterior=normalize_posterior,
            rescale_posterior=rescale_posterior,
            x_label=x_label,
            y_label=y_label,
            transposed=rotate,
        )

    if marginal_layout == "outside":
        for i, j in idcs_skipp:
            axs[i, j].set_visible(False)

    return plt.gcf()


@to.no_grad()
def draw_posterior_distr_pairwise_scatter(
    axs: plt.Axes,
    dp_samples: List[to.Tensor],
    dp_mapping: Mapping[int, str],
    marginal_layout: str = "outside",
    labels: Optional[List[str]] = None,
    legend_labels: Optional[List[str]] = None,
    prob_label: Optional[str] = None,
    color_palette=sns.color_palette(),
    set_alpha: Optional[float] = None,
    axis_limits: Optional[np.array] = None,
    use_kde: bool = False,
    custom_scatter_args: Optional[dict] = None,
    custom_histplot_args: Optional[dict] = None,
    custom_line_args: Optional[dict] = None,
    custom_kde_args: Optional[dict] = None,
) -> plt.Figure:
    """
    Plot a 2-dim gird of pairwise slices of the posterior distribution evaluated with samples from the posterior.

    :param axs: axis (joint) or axes (separately) of the figure to plot on
    :param dp_samples: domain parameter samples generated from different (posterior) distributions
    :param dp_mapping: mapping from subsequent integers (starting at 0) to domain parameter names (e.g. mass).
                       Here this mapping must not have more than 2 elements since we can't plot more.
    :param marginal_layout: choose between `inside` for plotting the marginals on the diagonal (more dense), and
                           `outside` plotting the marginals on the side (better comparison)
    :param labels: List of strings to set the axis labels, pass `None` to use the domain parameter mapping
    :param legend_labels: List of strings to set the legend labels, pass `None` to not use a legend
    :param prob_label: string to set the label for the probability axis in the marginal plots,
                       pass `""` to use default labels or pass `None` to use no labels
    :param color_palette: colorpalette for plotting the different distribution samples
    :param set_alpha: define a fixed alpha value for the scatter plot
    :param axis_limits: define the lower and upper limits for each dp as a numpy array. The shape is [2, num_dp]
                        passing `None` will generate limits automatically
    :param use_kde: set to True to plot samples with KDE (currently only for pair axes)
    :param custom_scatter_args: dict with additional settings for seaborn.scatterplot
    :param custom_histplot_args: dict with additional settings for seaborn.histplot
    :param custom_line_args: dict with additional settings for matplotlib.pyplot.vline and matplotlib.pyplot.hline
    :param custom_kde_args: dict with additional settings for seaborn.kdeplot
    :return: figure containing the pair plot
    """
    # Check the inputs
    if marginal_layout == "inside":
        plot_shape = (len(dp_mapping), len(dp_mapping))
    elif marginal_layout == "outside":
        plot_shape = (len(dp_mapping) + 1, len(dp_mapping) + 1)
    else:
        raise pyrado.ValueErr(given=marginal_layout, eq_constraint="inside or outside")
    assert isinstance(axs, np.ndarray)
    if axs.shape != plot_shape:
        raise pyrado.ShapeErr(given=axs, expected_match=plot_shape)

    if labels is None:
        # The default values for the labels are filled with the domain parameter names
        labels = [dp_mapping[k] for k in sorted(dp_mapping.keys())]
    elif type(labels) != list:
        raise pyrado.TypeErr(given=type(labels), expected_type=list)
    elif len(labels) != len(dp_mapping):
        raise pyrado.ShapeErr(given=labels, expected_match=dp_mapping)

    if legend_labels is not None:
        if type(legend_labels) != list:
            raise pyrado.TypeErr(given=type(legend_labels), expected_type=list)
        elif len(legend_labels) != len(dp_samples):
            raise pyrado.ShapeErr(given=legend_labels, expected_match=dp_samples)

    if prob_label == "":
        prob_label = "sample counts (KDE)" if use_kde else "sample counts"
    elif prob_label is None:
        pass
    else:
        prob_label = f"${prob_label}$"

    # Initialize plotting args and update them with custom args
    scatter_args = dict()
    scatter_args.update(custom_scatter_args or dict())
    histplot_args = dict(element="step", bins=50)
    histplot_args.update(custom_histplot_args or dict())
    line_args = dict(lw=2)
    line_args.update(custom_line_args or dict())
    kde_args = dict(fill=True)
    kde_args.update(custom_kde_args or dict())
    label_args = dict()  # fontsize=11

    # Generate the indices for the subplots
    if marginal_layout == "inside":
        # Set the indices for the marginal plots
        idcs_marginal = [idx for idx, _ in np.ndenumerate(axs) if idx[0] == idx[1]]
        # Set the indices for the pair plots
        idcs_pair = [idx for idx, _ in np.ndenumerate(axs) if idx not in idcs_marginal]
    elif marginal_layout == "outside":
        # Ignore the -1 lower diagonal and the single top right one
        idcs_skipp = [(i + 1, i) for i in range(plot_shape[0] - 1)] + [(0, plot_shape[1] - 1)]
        # Set the indices for the marginal plots
        idcs_marginal = [
            idx
            for idx, _ in np.ndenumerate(axs)
            if (idx[0] == 0 or idx[1] == plot_shape[1] - 1) and (idx not in idcs_skipp)
        ]
        # Set the indices for the pair plots
        idcs_pair = [idx for idx, _ in np.ndenumerate(axs) if idx not in idcs_marginal and idx not in idcs_skipp]

    assert check_all_types_equal(idcs_marginal) and check_all_lengths_equal(idcs_marginal)
    assert check_all_types_equal(idcs_pair) and check_all_lengths_equal(idcs_pair)

    # Apply settings for all axes
    for i, j in idcs_pair + idcs_marginal:
        # Rotate the ticklabels by 90° and format x- and y-axis
        plt.setp(axs[i, j].get_xticklabels(), rotation=90, horizontalalignment="center")
        axs[i, j].xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axs[i, j].xaxis.get_offset_text().set_rotation(90)
        axs[i, j].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # Define transparency values
    alphas = []
    for idx_obs in range(len(dp_samples)):
        if len(dp_samples[idx_obs]) == 1:
            alpha = 1.0
        else:
            alpha = 1 - idx_obs / len(dp_samples) if set_alpha is None else set_alpha
        alphas.append(alpha)

    # Plot data in pair axes
    for i, j in idcs_pair:
        plt.sca(axs[i, j])
        plt.minorticks_on()

        dim_x, dim_y = j, i  # dim_x, dim_y specify the data indices
        if marginal_layout == "outside":
            dim_y -= 1

        # Iterate through the data given sets
        for idx_obs in range(len(dp_samples)):
            # Plot the data points
            if use_kde and len(dp_samples[idx_obs]) > 1:
                sns.kdeplot(
                    x=dp_samples[idx_obs][:, dim_x],
                    y=dp_samples[idx_obs][:, dim_y],
                    alpha=alphas[idx_obs],
                    cmap=sns.light_palette(color_palette[idx_obs], as_cmap=True),
                    **kde_args,
                )
            else:
                sns.scatterplot(
                    x=dp_samples[idx_obs][:, dim_x],
                    y=dp_samples[idx_obs][:, dim_y],
                    color=color_palette[idx_obs],
                    alpha=alphas[idx_obs],
                    **scatter_args,
                )

    # Format pair axes. Set matplotlib axis limits based on the y-axis of the first column or cast them if were given.
    if axis_limits is None:
        i = 1 if marginal_layout == "outside" else 0
        axis_limits = [axs[i, 1].get_ylim()] + [axs[dim, 0].get_ylim() for dim in range(i + 1, plot_shape[0])]
        axis_limits = np.array(axis_limits).T
    else:
        axis_limits = axis_limits.numpy()

    for i, j in idcs_pair:
        dim_x, dim_y = j, i  # dim_x, dim_y specify the data indices
        if marginal_layout == "outside":
            dim_y -= 1

        # Update the axis limits for the pair_idcs
        axs[i, j].set_xlim(axis_limits[0, dim_x], axis_limits[1, dim_x])
        axs[i, j].set_ylim(axis_limits[0, dim_y], axis_limits[1, dim_y])

        if marginal_layout == "outside":
            # Show the ticklabels only for left and lower axes
            if i != plot_shape[0] - 1 and (i != plot_shape[0] - 2 or j != plot_shape[1] - 2):
                axs[i, j].set_xticklabels([])
            if j != 0 and (j != 1 or i != 1):
                axs[i, j].set_yticklabels([])

        if marginal_layout == "inside":
            if i == plot_shape[0] - 1 or (i == plot_shape[0] - 2 and j == plot_shape[1] - 1):
                axs[i, j].set_xlabel(f"${labels[dim_x]}$", **label_args)
            else:
                axs[i, j].set_xticklabels([])
            if j == 0 or (j == 1 and i == 0):
                axs[i, j].set_ylabel(f"${labels[dim_y]}$", **label_args)
            else:
                axs[i, j].set_yticklabels([])

    # Plot marginal axes
    max_marginal_y_limit = 0
    for i, j in idcs_marginal:
        plt.sca(axs[i, j])
        plt.minorticks_on()

        # dim_x, dim_y specify the data indices and decide if plot has to be rotated
        rotate = False
        if marginal_layout == "outside":
            dim = j if i == 0 else i - 1
            if i != 0:
                rotate = True
        elif marginal_layout == "inside":
            dim = i

        # Iterate through the given data sets
        for idx_obs in range(len(dp_samples)):
            # Update plotting arguments
            scatter_data = dp_samples[idx_obs][:, dim]  # current scatter data
            color = color_palette[idx_obs]
            current_histplot_args = dict(binrange=axis_limits[:, dim])
            current_histplot_args.update(histplot_args)
            current_kde_args = deepcopy(kde_args)

            # Define the settings depending on the rotation
            current_histplot_args.update(dict(y=scatter_data) if rotate else dict(x=scatter_data))
            current_kde_args.update(dict(y=scatter_data) if rotate else dict(x=scatter_data))
            line_obj = plt.axhline if rotate else plt.axvline

            # Plot a histogram or line depending on data size
            if len(scatter_data) == 1:
                line_obj(scatter_data, color=color, **line_args)
            elif use_kde:
                sns.kdeplot(color=color, alpha=alphas[idx_obs], **current_kde_args)
            else:
                sns.histplot(color=color, alpha=alphas[idx_obs], edgecolor=color, **current_histplot_args)

            if i == 0:
                max_marginal_y_limit = max(max_marginal_y_limit, axs[i, j].get_ylim()[1])

    # Format marginal axes
    for i, j in idcs_marginal:
        # dim_x, dim_y specify the data indices and decide if plot has to be rotated
        rotate = False
        if marginal_layout == "outside":
            dim = j if i == 0 else i - 1
            if i != 0:
                rotate = True
        elif marginal_layout == "inside":
            dim = i

        if rotate:
            axs[i, j].set_ylim(axis_limits[0, dim], axis_limits[1, dim])  # dp axis
            axs[i, j].set_xlim(0.0, max_marginal_y_limit)  # sample counts axis
        else:
            axs[i, j].set_xlim(axis_limits[0, dim], axis_limits[1, dim])
            axs[i, j].set_ylim(0.0, max_marginal_y_limit)

        # Set x-axis label and tick settings
        if marginal_layout == "outside":
            if i == 0:  # top row of axes
                axs[i, j].xaxis.set_label_position("top")  # set labels on top
                axs[i, j].set_xlabel(f"${labels[dim]}$", **label_args)
                axs[i, j].set_ylabel(prob_label if j == 0 else "", **label_args)
                axs[i, j].xaxis.tick_top()  # set ticks on top
                axs[i, j].set_xticklabels([])
                if j != 0:
                    axs[i, j].set_yticklabels([])

            if j == plot_shape[1] - 1:  # most right column of axes
                axs[i, j].yaxis.set_label_position("right")  # set labels to the right
                axs[i, j].set_ylabel(f"${labels[dim]}$", **label_args)
                axs[i, j].set_xlabel(prob_label if i == plot_shape[1] - 1 else "", **label_args)
                axs[i, j].yaxis.tick_right()  # set ticks to the right
                axs[i, j].set_yticklabels([])
                if i != plot_shape[1] - 1:
                    axs[i, j].set_xticklabels([])

        elif marginal_layout == "inside":
            axs[i, j].set_xlabel(f"${labels[dim]}$" if i == plot_shape[0] - 1 else "", **label_args)
            axs[i, j].set_ylabel(prob_label, **label_args)
            if i != plot_shape[0] - 1:
                axs[i, j].set_xticklabels([])

    # Format other axes
    if marginal_layout == "outside":
        for i, j in idcs_skipp:
            # Hide diagonal axes for outside layout
            axs[i, j].set_visible(False)

    # Build legend
    if legend_labels is not None:
        legend_elements = list()
        for idx_obs in range(len(dp_samples)):
            legend_elements.append(Patch(facecolor=color_palette[idx_obs], label=legend_labels[idx_obs]))

        # Create the legend
        if marginal_layout == "outside":
            ax_leg = axs[0, plot_shape[1] - 1]
            ax_leg.set_visible(True)
            ax_leg.axis("off")
            ax_leg.legend(handles=legend_elements, loc="center")
        elif marginal_layout == "inside":
            fig = plt.gcf()
            fig.legend(handles=legend_elements, loc=(0.5, 1), ncol=len(legend_elements))

    return plt.gcf()
