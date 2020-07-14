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
from matplotlib import pyplot as plt


def plot_dts(dts_policy: np.ndarray, dts_step: np.ndarray, dts_remainder: np.ndarray):
    r"""
    Plot the time intervals ($\Delta_t$) of various parts of one time step.

    :param dts_policy: time it took to compute the policy's action
    :param dts_step: time it took to perform the
    :param dts_remainder: time it took to execute all remaining commands (e.g. soring the data)
    """
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), tight_layout=True)

    x = np.arange(0, len(dts_policy))
    y1 = 1000*dts_policy
    y2 = 1000*dts_step
    y3 = 1000*dts_remainder
    labels = [r'dt\_policy [ms]', r'dt\_step [ms]', r'dt\_remainder [ms]']

    axs[0].plot(x, y1, label=labels[0])
    axs[0].plot(x, y2, label=labels[1])
    axs[0].plot(x, y3, label=labels[2])
    axs[0].axhline(y=2, color='k')  # at 2 ms we exceed the sampling frequency of 500 Hz
    # axs[0].set_ylim(top=5)  # limit to 5 ms
    axs[0].legend(loc='upper right')
    axs[0].set_title('individual plots')

    axs[1].stackplot(x, y1, y2, y3, labels=labels)
    axs[1].axhline(y=2, color='k')  # at 2 ms we exceed the sampling frequency of 500 Hz
    # axs[1].set_ylim(top=5)  # limit to 5 ms
    axs[1].legend(loc='upper right')
    axs[1].set_title('stack plot')

    plt.show()


def render_lo_up_avg(
    ax: plt.Axes,
    x_grid: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    average: np.ndarray,
    x_label: str,
    y_label: str,
    curve_label: str,
    alpha: float = 0.3,
    color: chr = None,
    show_legend: bool = True,
    area_label: [str, None] = r'min \& max',
    title: str = None,
):
    """
    Plot the given average, minimum, and maximum values over given x-axis data. The plot is neither shown nor saved.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param ax: axis of the figure to plot on
    :param x_grid: data to plot on the x-axis
    :param lower: minimum values to plot on the y-axis
    :param upper: maximum values to plot on the y-axis
    :param average: average values to plot on the y-axis
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param curve_label: label for the average curve
    :param alpha: transparency (alpha-value) for the std area
    :param color: color (e.g. 'k'), None invokes the default behavior
    :param show_legend: flag if the legend entry should be printed, set to True when using multiple subplots
    :param area_label: label for the shaded area, pass None to omit the label
    :param title: title displayed above the figure, set to None to suppress the title
    :param tight_layout: if True, the x and y axes will have no space to the plotted curves
    :return: handle to the resulting figure
    """
    if area_label is not None:
        ax.fill_between(x_grid, lower, upper, alpha=alpha, label=area_label)
    else:
        ax.fill_between(x_grid, lower, upper, alpha=alpha)
    ax.plot(x_grid, average, label=curve_label, color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    return plt.gcf()


def render_mean_std(
    ax: plt.Axes,
    x_grid: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    x_label: str,
    y_label: str,
    curve_label: str,
    num_stds: int = 1,
    alpha: float = 0.3,
    color: chr = None,
    show_legend: bool = True,
    show_legend_std: bool = False,
    title: str = None,
) -> plt.Figure:
    """
    Plot the given mean and the standard deviation over given x-axis data. The plot is neither shown nor saved.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param ax: axis of the figure to plot on
    :param x_grid: data to plot on the x-axis
    :param mean: mean values to plot on the y-axis
    :param std: standard deviation values to plot on the y-axis
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param curve_label: label for the mean curve
    :param num_stds: number of standard deviations to plot around the mean
    :param alpha: transparency (alpha-value) for the std area
    :param color: color (e.g. 'k'), None invokes the default behavior
    :param show_legend: flag if the legend entry should be printed, set to True when using multiple subplots
    :param show_legend_std: flag if a legend entry for the std area should be printed
    :param title: title displayed above the figure, set to None to suppress the title
    :param tight_layout: if True, the x and y axes will have no space to the plotted curves
    :return: handle to the resulting figure
    """
    ax.plot(x_grid, mean, label=curve_label, color=color)
    if show_legend_std:
        ax.fill_between(x_grid, mean - num_stds*std, mean + num_stds*std, alpha=alpha, label=f'$\pm {num_stds}$ std')
    else:
        ax.fill_between(x_grid, mean - num_stds*std, mean + num_stds*std, alpha=alpha)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend()
    if title is not None:
        ax.set_title(title)
    return plt.gcf()


def render_data_mean_std(
    ax: plt.Axes,
    x_grid: np.ndarray,
    data: np.ndarray,
    x_label: str,
    y_label: str,
    curve_label: str,
    ax_calc: int = 0,
    num_stds: int = 1,
    alpha: float = 0.3,
    color: chr = None,
    show_legend: bool = True,
    show_legend_std: bool = False,
    title: str = None,
) -> plt.Figure:
    """
    Compute and plot the mean and the standard deviation of the given data. The plot is neither shown nor saved.

    :param ax: axis of the figure to plot on
    :param x_grid: data to plot on the x-axis
    :param data: data to process and plot on the y-axis
    :param x_label: label for x-axis
    :param y_label: label for y-axis
    :param curve_label: label for the mean curve
    :param ax_calc: axis of the data array to calculate the mean over
    :param num_stds: number of standard deviations to plot around the mean
    :param alpha: transparency (alpha-value) for the std area
    :param color: color (e.g. 'k'), None invokes the default behavior
    :param show_legend: flag if the legend entry should be printed, set to True when using multiple subplots
    :param show_legend_std: flag if a legend entry for the std area should be printed
    :param title: title displayed above the figure, set to None to suppress the title
    :return: handle to the resulting figure
    """
    # Calculate mean and std values along the given axis
    mean = np.mean(data, axis=ax_calc)
    std = np.std(data, axis=ax_calc)

    # Plot the data
    return render_mean_std(ax, x_grid, mean, std, x_label, y_label, curve_label,
                           num_stds=num_stds, alpha=alpha, color=color, show_legend_std=show_legend_std,
                           show_legend=show_legend, title=title)
