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

from typing import Sequence

import numpy as np
import pandas as pd
import torch as to
from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import GreaterThan
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib import colors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyrado
from pyrado.plotting.heatmap import draw_heatmap
from pyrado.utils.input_output import print_cbt


def render_singletask_gp(
    ax: [plt.Axes, Axes3D, Sequence[plt.Axes]],
    data_x: to.Tensor,
    data_y: to.Tensor,
    idcs_sel: list,
    data_x_min: to.Tensor = None,
    data_x_max: to.Tensor = None,
    x_label: str = "",
    y_label: str = "",
    z_label: str = "",
    min_gp_obsnoise: float = None,
    resolution: int = 201,
    num_stds: int = 2,
    alpha: float = 0.3,
    color: chr = None,
    heatmap_cmap: colors.Colormap = None,
    show_legend_posterior: bool = True,
    show_legend_data: bool = True,
    legend_data_cmap: colors.Colormap = None,
    colorbar_label: str = None,
    render3D: bool = True,
) -> plt.Figure:
    """
    Fit the GP posterior to the input data and plot the mean and std as well as the data points.
    There are 3 options: 1D plot (infered by data dimensions), 2D plot

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param ax: axis of the figure to plot on, only in case of a 2-dim heat map plot provide 2 axis
    :param data_x: data to plot on the x-axis
    :param data_y: data to process and plot on the y-axis
    :param idcs_sel: selected indices of the input data
    :param data_x_min: explicit minimum value for the evaluation grid, by default this value is extracted from `data_x`
    :param data_x_max: explicit maximum value for the evaluation grid, by default this value is extracted from `data_x`
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param z_label: label for z-axis (3D plot only)
    :param min_gp_obsnoise: set a minimal noise value (normalized) for the GP, if `None` the GP has no measurement noise
    :param resolution: number of samples for the input (corresponds to x-axis resolution of the plot)
    :param num_stds: number of standard deviations to plot around the mean
    :param alpha: transparency (alpha-value) for the std area
    :param color: color (e.g. 'k' for black), `None` invokes the default behavior
    :param heatmap_cmap: color map forwarded to `draw_heatmap()` (2D plot only), `None` to use Pyrado's default
    :param show_legend_posterior: flag if the legend entry for the posterior should be printed (affects mean and std)
    :param show_legend_data: flag if a legend entry for the individual data points should be printed
    :param legend_data_cmap: color map for the sampled points, default is 'binary'
    :param colorbar_label: label for the color bar (2D plot only)
    :param render3D: use 3D rendering if possible
    :return: handle to the resulting figure
    """
    if data_x.ndim != 2:
        raise pyrado.ShapeErr(msg="The GP's input data needs to be of shape num_samples x dim_input!")
    data_x = data_x[:, idcs_sel]  # forget the rest
    dim_x = data_x.shape[1]  # samples are along axis 0

    if data_y.ndim != 2:
        raise pyrado.ShapeErr(given=data_y, expected_match=to.Size([data_x.shape[0], 1]))

    if legend_data_cmap is None:
        legend_data_cmap = plt.get_cmap("binary")

    # Project to normalized input and standardized output
    if data_x_min is None or data_x_max is None:
        data_x_min, data_x_max = to.min(data_x, dim=0)[0], to.max(data_x, dim=0)[0]
    data_y_mean, data_y_std = to.mean(data_y, dim=0), to.std(data_y, dim=0)
    data_x = (data_x - data_x_min) / (data_x_max - data_x_min)
    data_y = (data_y - data_y_mean) / data_y_std

    # Create and fit the GP model
    gp = SingleTaskGP(data_x, data_y)
    if min_gp_obsnoise is not None:
        gp.likelihood.noise_covar.register_constraint("raw_noise", GreaterThan(min_gp_obsnoise))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    mll.train()
    fit_gpytorch_model(mll)
    print_cbt("Fitted the SingleTaskGP.", "g")

    argmax_pmean_norm, argmax_pmean_val_stdzed = optimize_acqf(
        acq_function=PosteriorMean(gp),
        bounds=to.stack([to.zeros(dim_x), to.ones(dim_x)]),
        q=1,
        num_restarts=500,
        raw_samples=1000,
    )
    # Project back
    argmax_posterior = argmax_pmean_norm * (data_x_max - data_x_min) + data_x_min
    argmax_pmean_val = argmax_pmean_val_stdzed * data_y_std + data_y_mean
    print_cbt(f"Converged to argmax of the posterior mean: {argmax_posterior.numpy()}", "g")

    mll.eval()
    gp.eval()

    if dim_x == 1:
        # Evaluation grid
        x_grid = np.linspace(min(data_x), max(data_x), resolution, endpoint=True).flatten()
        x_grid = to.from_numpy(x_grid)

        # Mean and standard deviation of the surrogate model
        posterior = gp.posterior(x_grid)
        mean = posterior.mean.detach().flatten()
        std = to.sqrt(posterior.variance.detach()).flatten()

        # Project back from normalized input and standardized output
        x_grid = x_grid * (data_x_max - data_x_min) + data_x_min
        data_x = data_x * (data_x_max - data_x_min) + data_x_min
        data_y = data_y * data_y_std + data_y_mean
        mean = mean * data_y_std + data_y_mean
        std *= data_y_std  # double-checked with posterior.mvn.confidence_region()

        # Plot the curve
        plt.fill_between(
            x_grid.numpy(),
            mean.numpy() - num_stds * std.numpy(),
            mean.numpy() + num_stds * std.numpy(),
            alpha=alpha,
            color=color,
        )
        ax.plot(x_grid.numpy(), mean.numpy(), color=color)

        # Plot the queried data points
        scat_plot = ax.scatter(
            data_x.numpy().flatten(),
            data_y.numpy().flatten(),
            marker="o",
            c=np.arange(data_x.shape[0], dtype=np.int),
            cmap=legend_data_cmap,
        )

        if show_legend_data:
            scat_legend = ax.legend(
                *scat_plot.legend_elements(fmt="{x:.0f}"),  # integer formatter
                bbox_to_anchor=(0.0, 1.1, 1.0, -0.1),
                title="query points",
                ncol=data_x.shape[0],
                loc="upper center",
                mode="expand",
                borderaxespad=0.0,
                handletextpad=-0.5,
            )
            ax.add_artist(scat_legend)
            # Increase vertical space between subplots when printing the data labels
            # plt.tight_layout(pad=2.)  # ignore argument
            # plt.subplots_adjust(hspace=0.6)

        # Plot the argmax of the posterior mean
        # ax.scatter(argmax_posterior.item(), argmax_pmean_val, c='darkorange', marker='o', s=60, label='argmax')
        ax.axvline(argmax_posterior.item(), c="darkorange", lw=1.5, label="argmax")

        if show_legend_posterior:
            ax.add_artist(ax.legend(loc="lower right"))

    elif dim_x == 2:
        # Create mesh grid matrices from x and y vectors
        # x0_grid = to.linspace(min(data_x[:, 0]), max(data_x[:, 0]), resolution)
        # x1_grid = to.linspace(min(data_x[:, 1]), max(data_x[:, 1]), resolution)
        x0_grid = to.linspace(0, 1, resolution)
        x1_grid = to.linspace(0, 1, resolution)
        x0_mesh, x1_mesh = to.meshgrid([x0_grid, x1_grid])
        x0_mesh, x1_mesh = x0_mesh.t(), x1_mesh.t()  # transpose not necessary but makes identical mesh as np.meshgrid

        # Mean and standard deviation of the surrogate model
        x_test = to.stack([x0_mesh.reshape(resolution ** 2, 1), x1_mesh.reshape(resolution ** 2, 1)], -1).squeeze(1)
        posterior = gp.posterior(x_test)  # identical to  gp.likelihood(gp(x_test))
        mean = posterior.mean.detach().reshape(resolution, resolution)
        std = to.sqrt(posterior.variance.detach()).reshape(resolution, resolution)

        # Project back from normalized input and standardized output
        data_x = data_x * (data_x_max - data_x_min) + data_x_min
        data_y = data_y * data_y_std + data_y_mean
        mean_raw = mean * data_y_std + data_y_mean
        std_raw = std * data_y_std

        if render3D:
            # Project back from normalized input and standardized output (custom for 3D)
            x0_mesh = x0_mesh * (data_x_max[0] - data_x_min[0]) + data_x_min[0]
            x1_mesh = x1_mesh * (data_x_max[1] - data_x_min[1]) + data_x_min[1]
            lower = mean_raw - num_stds * std_raw
            upper = mean_raw + num_stds * std_raw

            # Plot a 2D surface in 3D
            ax.plot_surface(x0_mesh.numpy(), x1_mesh.numpy(), mean_raw.numpy())
            ax.plot_surface(x0_mesh.numpy(), x1_mesh.numpy(), lower.numpy(), color="r", alpha=alpha)
            ax.plot_surface(x0_mesh.numpy(), x1_mesh.numpy(), upper.numpy(), color="r", alpha=alpha)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)

            # Plot the queried data points
            scat_plot = ax.scatter(
                data_x[:, 0].numpy(),
                data_x[:, 1].numpy(),
                data_y.numpy(),
                marker="o",
                c=np.arange(data_x.shape[0], dtype=np.int),
                cmap=legend_data_cmap,
            )

            if show_legend_data:
                scat_legend = ax.legend(
                    *scat_plot.legend_elements(fmt="{x:.0f}"),  # integer formatter
                    bbox_to_anchor=(0.05, 1.1, 0.95, -0.1),
                    loc="upper center",
                    ncol=data_x.shape[0],
                    mode="expand",
                    borderaxespad=0.0,
                    handletextpad=-0.5,
                )
                ax.add_artist(scat_legend)

            # Plot the argmax of the posterior mean
            x, y = argmax_posterior[0, 0], argmax_posterior[0, 1]
            ax.scatter(x, y, argmax_pmean_val, c="darkorange", marker="*", s=60)
            # ax.plot((x, x), (y, y), (data_y.min(), data_y.max()), c='k', ls='--', lw=1.5)

        else:
            if not len(ax) == 4:
                raise pyrado.ShapeErr(msg="Provide 4 axes! 2 heat maps and 2 color bars.")

            # Project back normalized input and standardized output (custom for 2D)
            x0_grid_raw = x0_grid * (data_x_max[0] - data_x_min[0]) + data_x_min[0]
            x1_grid_raw = x1_grid * (data_x_max[1] - data_x_min[1]) + data_x_min[1]

            # Plot a 2D image
            df_mean = pd.DataFrame(mean_raw.numpy(), columns=x0_grid_raw.numpy(), index=x1_grid_raw.numpy())
            draw_heatmap(
                df_mean,
                ax_hm=ax[0],
                ax_cb=ax[1],
                x_label=x_label,
                y_label=y_label,
                annotate=False,
                fig_canvas_title="Returns",
                tick_label_prec=2,
                separate_cbar=True,
                cmap=heatmap_cmap,
                colorbar_label=colorbar_label,
                num_major_ticks_hm=3,
                num_major_ticks_cb=2,
                colorbar_orientation="horizontal",
            )

            df_std = pd.DataFrame(std_raw.numpy(), columns=x0_grid_raw.numpy(), index=x1_grid_raw.numpy())
            draw_heatmap(
                df_std,
                ax_hm=ax[2],
                ax_cb=ax[3],
                x_label=x_label,
                y_label=y_label,
                annotate=False,
                fig_canvas_title="Standard Deviations",
                tick_label_prec=2,
                separate_cbar=True,
                cmap=heatmap_cmap,
                colorbar_label=colorbar_label,
                num_major_ticks_hm=3,
                num_major_ticks_cb=2,
                colorbar_orientation="horizontal",
                norm=colors.Normalize(),
            )  # explicitly instantiate a new norm

            # Plot the queried data points
            for i in [0, 2]:
                scat_plot = ax[i].scatter(
                    data_x[:, 0].numpy(),
                    data_x[:, 1].numpy(),
                    marker="o",
                    s=15,
                    c=np.arange(data_x.shape[0], dtype=np.int),
                    cmap=legend_data_cmap,
                )

                if show_legend_data:
                    scat_legend = ax[i].legend(
                        *scat_plot.legend_elements(fmt="{x:.0f}"),  # integer formatter
                        bbox_to_anchor=(0.0, 1.1, 1.0, 0.05),
                        loc="upper center",
                        ncol=data_x.shape[0],
                        mode="expand",
                        borderaxespad=0.0,
                        handletextpad=-0.5,
                    )
                    ax[i].add_artist(scat_legend)

            # Plot the argmax of the posterior mean
            ax[0].scatter(argmax_posterior[0, 0], argmax_posterior[0, 1], c="darkorange", marker="*", s=60)  # steelblue
            ax[2].scatter(argmax_posterior[0, 0], argmax_posterior[0, 1], c="darkorange", marker="*", s=60)  # steelblue
            # ax[0].axvline(argmax_posterior[0, 0], c='w', ls='--', lw=1.5)
            # ax[0].axhline(argmax_posterior[0, 1], c='w', ls='--', lw=1.5)
            # ax[2].axvline(argmax_posterior[0, 0], c='w', ls='--', lw=1.5)
            # ax[2].axhline(argmax_posterior[0, 1], c='w', ls='--', lw=1.5)

    else:
        raise pyrado.ValueErr(msg="Can only plot 1-dim or 2-dim data!")

    return plt.gcf()
