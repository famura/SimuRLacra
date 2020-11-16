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
import pandas as pd
import torch as to
from matplotlib import pyplot as plt
from typing import Union, Sequence, Optional

import pyrado
from pyrado.utils.data_types import merge_dicts


def draw_dts(dts_policy: np.ndarray, dts_step: np.ndarray, dts_remainder: np.ndarray):
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
    labels = [r'$\Delta$t policy [ms]', r'$\Delta$t step [ms]', r'$\Delta$t remainder [ms]']

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


def draw_curve(
    plot_type: str,
    ax: plt.Axes,
    data: Union[list, np.ndarray, to.Tensor, pd.DataFrame],
    x_grid: Union[list, np.ndarray, to.Tensor],
    ax_calc: int,
    x_label: Optional[Union[str, Sequence[str]]],
    y_label: Optional[str],
    curve_label: Optional[str] = None,
    area_label: Optional[str] = None,
    vline_level: float = None,
    vline_label: str = 'approx. solved',
    palette=None,
    title: str = None,
    show_legend: bool = True,
    legend_kwargs: dict = None,
    plot_kwargs: dict = None
) -> plt.Figure:
    """
    Create a box or violin plot for a list of data arrays or a pandas DataFrame.
    The plot is neither shown nor saved.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

        If you want to order the 4th element to the 2nd position in terms of colors use
        .. code-block:: python

            palette.insert(1, palette.pop(3))

    :param plot_type: tye of categorical plot, pass box or violin
    :param ax: axis of the figure to plot on
    :param data: list of data sets to plot as separate boxes
    :param x_grid:
    :param ax_calc: axis of the data array to calculate the mean, min and max, or std over
    :param x_label: labels for the categories on the x-axis, if `data` is not given as a `DataFrame`
    :param y_label: label for the y-axis, pass `None` to set no label
    :param curve_label: label of the (1-dim) curve
    :param area_label: label of the (transparent) area
    :param vline_level: if not `None` (default) add a vertical line at the given level
    :param vline_label: label for the vertical line
    :param palette: seaborn color palette, pass `None` to use the default palette
    :param show_legend: if `True` the legend is shown, useful when handling multiple subplots
    :param title: title displayed above the figure, set to None to suppress the title
    :param legend_kwargs: keyword arguments forwarded to pyplot's `legend()` function, e.g. `loc='best'`
    :param plot_kwargs: keyword arguments forwarded to seaborn's `boxplot()` or `violinplot()` function
    :return: handle to the resulting figure
    """
    plot_type = plot_type.lower()
    if plot_type not in ['mean_std', 'min_mean_max']:
        raise pyrado.ValueErr(given=plot_type, eq_constraint='mean_std or min_mean_max')
    if not isinstance(data, (list, to.Tensor, np.ndarray, pd.DataFrame)):
        raise pyrado.TypeErr(given=data, expected_type=[list, to.Tensor, np.ndarray, pd.DataFrame])
    if x_label is not None and not isinstance(x_label, str):
        raise pyrado.TypeErr(given=x_label, expected_type=str)
    if y_label is not None and not isinstance(y_label, str):
        raise pyrado.TypeErr(given=y_label, expected_type=str)

    # Set defaults which can be overwritten by passing plot_kwargs
    plot_kwargs = merge_dicts([dict(alpha=0.3), plot_kwargs])
    legend_kwargs = dict() if legend_kwargs is None else legend_kwargs
    # palette = sns.color_palette() if palette is None else palette

    # Preprocess
    if isinstance(x_grid, list):
        x_grid = np.array(x_grid)
    elif isinstance(x_grid, to.Tensor):
        x_grid = x_grid.detach().cpu().numpy()

    if isinstance(data, pd.DataFrame):
        # df = data.copy()
        data = data.to_numpy()
    else:
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, to.Tensor):
            data = data.detach().cpu().numpy()
        # df = pd.DataFrame(data)

    # Plot
    mean = np.mean(data, axis=ax_calc)
    if plot_type == 'mean_std':
        num_stds = 2
        if area_label is None:
            area_label = rf'$\pm {num_stds}$ std'
        std = np.std(data, axis=ax_calc)
        ax.fill_between(x_grid, mean - num_stds*std, mean + num_stds*std, label=area_label, **plot_kwargs)

    elif plot_type == 'min_mean_max':
        if area_label is None:
            area_label = 'min max'
        lower = np.min(data, axis=ax_calc)
        upper = np.max(data, axis=ax_calc)
        ax.fill_between(x_grid, lower, upper, label=area_label, **plot_kwargs)

    # plot mean last for proper z-ordering
    plot_kwargs['alpha'] = 1
    ax.plot(x_grid, mean, label=curve_label, **plot_kwargs)

    # Postprocess
    if vline_level is not None:
        # Add dashed line to mark a threshold
        ax.axhline(vline_level, c='k', ls='--', lw=1., label=vline_label)

    if x_label is None:
        ax.get_xaxis().set_ticks([])

    if y_label is not None:
        ax.set_ylabel(y_label)

    if show_legend:
        ax.legend(**legend_kwargs)

    if title is not None:
        ax.set_title(title)

    return plt.gcf()
