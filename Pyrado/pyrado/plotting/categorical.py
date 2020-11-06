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
import seaborn as sns
import torch as to
from matplotlib import pyplot as plt
from typing import Sequence, Union, Optional

import pyrado
from pyrado.utils.data_types import merge_dicts
from pyrado.utils.input_output import print_cbt


def draw_categorical(
    plot_type: str,
    ax: plt.Axes,
    data: Union[list, np.ndarray, to.Tensor, pd.DataFrame],
    x_label: Optional[Union[str, Sequence[str]]],
    y_label: Optional[str],
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
    :param x_label: labels for the categories on the x-axis, if `data` is not given as a `DataFrame`
    :param y_label: label for the y-axis, pass `None` to set no label
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
    if plot_type not in ['box', 'violin']:
        raise pyrado.ValueErr(given=plot_type, eq_constraint='box or violin')
    if not isinstance(data, (list, to.Tensor, np.ndarray, pd.DataFrame)):
        raise pyrado.TypeErr(given=data, expected_type=[list, to.Tensor, np.ndarray, pd.DataFrame])

    # Set defaults which can be overwritten
    plot_kwargs = merge_dicts([dict(alpha=1), plot_kwargs])  # by default no transparency
    alpha = plot_kwargs.pop('alpha')  # can't pass the to the seaborn plotting functions
    legend_kwargs = dict() if legend_kwargs is None else legend_kwargs
    palette = sns.color_palette() if palette is None else palette

    # Preprocess
    if isinstance(data, pd.DataFrame):
        df = data
    else:
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, to.Tensor):
            data = data.detach().cpu().numpy()
        if x_label is not None and not len(x_label) == data.shape[1]:
            raise pyrado.ShapeErr(given=data, expected_match=x_label)
        df = pd.DataFrame(data, columns=x_label)

    if data.shape[0] < data.shape[1]:
        print_cbt(f'Less data samples {data.shape[0]} then data dimensions {data.shape[1]}', 'y', bright=True)

    # Plot
    if plot_type == 'box':
        ax = sns.boxplot(data=df, ax=ax, **plot_kwargs)

    elif plot_type == 'violin':
        plot_kwargs = merge_dicts([dict(alpha=0.3, scale='count', inner='box', bw=0.3, cut=0), plot_kwargs])
        ax = sns.violinplot(data=df, ax=ax, palette=palette, **plot_kwargs)

        # Plot larger circles for medians (need to memorize the limits)
        medians = df.median().to_numpy()
        left, right = ax.get_xlim()
        locs = ax.get_xticks()
        ax.scatter(locs, medians, marker='o', s=30, zorder=3, color='white', edgecolors='black')
        ax.set_xlim((left, right))

    # Postprocess
    if alpha < 1 and plot_type == 'box':
        for patch in ax.artists:
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, alpha))
    elif alpha < 1 and plot_type == 'violin':
        for violin in ax.collections[::2]:
            violin.set_alpha(alpha)

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
