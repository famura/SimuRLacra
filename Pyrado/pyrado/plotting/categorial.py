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
from typing import Sequence


def render_boxplot(
    ax: plt.Axes,
    data: [Sequence[list], Sequence[np.ndarray]],
    x_labels: Sequence[str],
    y_label: str,
    vline_level: float = None,
    vline_label: str = 'approx. solved',
    alpha: float = 1.,
    colorize: bool = False,
    show_fliers: bool = False,
    show_legend: bool = True,
    legend_loc: str = 'best',
    title: str = None,
    use_seaborn: bool = False,
) -> plt.Figure:
    """
    Create a box plot for a list of data arrays. Every entry results in one column of the box plot.
    The plot is neither shown nor saved.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param ax: axis of the figure to plot on
    :param data: list of data sets to plot as separate boxes
    :param x_labels: labels for the categories on the x-axis
    :param y_label: label for the y-axis
    :param vline_level: if not `None` (default) add a vertical line at the given level
    :param vline_label: label for the vertical line
    :param alpha: transparency (alpha-value) for boxes (including the border lines)
    :param colorize: colorize the core of the boxes
    :param show_fliers: show outliers (more the 1.5 of the inter quartial range) as circles
    :param show_legend: flag if the legend entry should be printed, set to True when using multiple subplots
    :param legend_loc: location of the legend, ignored if `show_legend = False`
    :param title: title displayed above the figure, set to None to suppress the title
    :param use_seaborn: if `True`, use seaborn instead of plain pyplot
    :return: handle to the resulting figure
    """
    if use_seaborn:
        # Plot the data
        import seaborn as sns
        import pandas as pd

        df = pd.DataFrame(data, x_labels).T
        ax = sns.boxplot(data=df)

    else:
        medianprops = dict(linewidth=1., color='firebrick')
        meanprops = dict(marker='D', markeredgecolor='black', markerfacecolor='purple')
        boxprops = dict(linewidth=1.)
        whiskerprops = dict(linewidth=1.)
        capprops = dict(linewidth=1.)

        # Plot the data
        box = ax.boxplot(
            data,
            boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops,
            meanprops=meanprops, meanline=False, showmeans=False,
            medianprops=medianprops,
            showfliers=show_fliers,
            notch=False,
            patch_artist=colorize,  # necessary to colorize the boxes
            labels=x_labels, widths=0.7
        )

        if colorize:
            for i, patch in enumerate(box['boxes']):
                patch.set_facecolorf(f'C{i%10}')
                patch.set_alpha(alpha)

    # Add dashed line to mark the approx solved threshold
    if vline_level is not None:
        ax.axhline(vline_level, c='k', ls='--', lw=1., label=vline_label)

    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend(loc=legend_loc)
    if title is not None:
        ax.set_title(title)
    return plt.gcf()


def render_violinplot(
    ax: plt.Axes,
    data: [Sequence[list], Sequence[np.ndarray]],
    x_labels: Sequence[str],
    y_label: str,
    vline_level: float = None,
    vline_label: str = 'approx. solved',
    alpha: float = 0.7,
    show_inner_quartiles: bool = False,
    show_legend: bool = True,
    legend_loc: str = 'best',
    title: str = None,
    use_seaborn: bool = False,
) -> plt.Figure:
    """
    Create a violin plot for a list of data arrays. Every entry results in one column of the violin plot.
    The plot is neither shown nor saved.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param ax: axis of the figure to plot on
    :param data: list of data sets to plot as separate violins
    :param x_labels: labels for the categories on the x-axis
    :param y_label: label for the y-axis
    :param vline_level: if not `None` (default) add a vertical line at the given level
    :param vline_label: label for the vertical line
    :param alpha: transparency (alpha-value) for violin body (including the border lines)
    :param show_inner_quartiles: display the 1st and 3rd quartile with a thick line, ignored if `use_seaborn` is `True`
    :param show_legend: flag if the legend entry should be printed, set to `True` when using multiple subplots
    :param legend_loc: location of the legend, ignored if `show_legend = False`
    :param title: title displayed above the figure, set to None to suppress the title
    :param use_seaborn: if `True`, use seaborn instead of plain pyplot
    :return: handle to the resulting figure
    """
    if use_seaborn:
        # Plot the data
        import seaborn as sns
        import pandas as pd

        # Get default color map and move the 4th element to the 2nd position
        palette = sns.color_palette()[:4]
        palette.insert(1, palette.pop(3))

        df = pd.DataFrame(data, x_labels).T
        ax = sns.violinplot(data=df, scale='count', inner='box', bw=0.3, cut=0, palette=palette)
        # ax = sns.violinplot(data=df, scale='width', inner='quartile', bw=0.3, cut=0)  # cut controls the max7min values

        # Grouped boxplot
        # df = pd.DataFrame(columns=['algo', 'succ_rate', 'domain'])
        # df = pd.concat([df, pd.DataFrame.from_dict(dict(algo='BayRn', succ_rate=data[0], domain='real'))])
        # df = pd.concat([df, pd.DataFrame.from_dict(dict(algo='BayRn', succ_rate=data[1], domain='sim'))])
        # df = pd.concat([df, pd.DataFrame.from_dict(dict(algo='UDR', succ_rate=data[2], domain='real'))])
        # df = pd.concat([df, pd.DataFrame.from_dict(dict(algo='UDR', succ_rate=data[3], domain='sim'))])
        # df = pd.concat([df, pd.DataFrame.from_dict(dict(algo='PPO', succ_rate=data[4], domain='real'))])
        # df = pd.concat([df, pd.DataFrame.from_dict(dict(algo='PPO', succ_rate=data[5], domain='sim'))])
        # ax = sns.violinplot(data=df, scale='width', bw=0.3, cut=0,
        #                     palette="Set1", x='algo', y='succ_rate', hue='domain', hue_order=['real', 'sim'])

        medians = np.zeros(len(data))
        for i in range(len(data)):
            medians[i] = np.median(data[i])

        # Plot larger circles
        left, right = ax.get_xlim()
        locs = ax.get_xticks()
        ax.scatter(locs, medians, marker='o', s=30, zorder=3, color='white', edgecolors='black')
        ax.set_xlim((left, right))

    else:
        # Plot the data
        violin = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)

        # Set custom color scheme
        for pc in violin['bodies']:
            pc.set_facecolor('#b11226')
            pc.set_edgecolor('black')
            pc.set_alpha(alpha)

        # Set axis style
        ax.set_xticks(np.arange(1, len(x_labels) + 1))
        ax.set_xticklabels(x_labels)

        quartiles_up, medians, quartiles_lo = np.zeros(len(data)), np.zeros(len(data)), np.zeros(len(data))
        data_mins, data_maxs = np.zeros(len(data)), np.zeros(len(data))
        for i in range(len(data)):
            quartiles_up[i], medians[i], quartiles_lo[i] = np.percentile(data[i], [25, 50, 75])
            data_mins[i], data_maxs[i] = min(data[i]), max(data[i])

        x_grid = np.arange(1, len(medians) + 1)
        ax.scatter(x_grid, medians, marker='o', s=50, zorder=3, color='white', edgecolors='black')
        ax.vlines(x_grid, data_mins, data_maxs, color='k', linestyle='-', lw=1, alpha=alpha)
        if show_inner_quartiles:
            ax.vlines(x_grid, quartiles_up, quartiles_lo, color='k', linestyle='-', lw=5)

    # Add dashed line to mark the approx solved threshold
    if vline_level is not None:
        ax.axhline(vline_level, c='k', ls='--', lw=1.0, label=vline_label)

    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend(loc=legend_loc)
    if title is not None:
        ax.set_title(title)
    return plt.gcf()
