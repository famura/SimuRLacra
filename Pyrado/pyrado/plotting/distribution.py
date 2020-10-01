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
import torch as to
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from torch.distributions import Distribution
from typing import Sequence, Iterable

import pyrado
from pyrado.utils.checks import check_all_types_equal


def render_distr_evo(
    ax: plt.Axes,
    distributions: Iterable,
    x_grid_limits: [tuple, list, np.ndarray],
    x_label: str = '',
    y_label: str = '',
    distr_labels: Sequence[str] = None,
    resolution: int = 201,
    alpha: float = 0.3,
    cmap_name: str = 'plasma',
    show_legend: bool = True,
    title: str = None,
) -> plt.Figure:
    """
    Plot the evolution of a sequence of PyTorch probability distributions.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param ax: axis of the figure to plot on
    :param distributions: iterable with the distributions in the order they should be plotted
    :param x_grid_limits: min and max value for the evaluation grid
    :param x_label: label for the x-axis
    :param y_label: label for the y-axis
    :param distr_labels: label for each of the distributions
    :param resolution: number of samples for the input (corresponds to x-axis resolution of the plot)
    :param cmap_name: name of the color map, e.g. 'inferno', 'RdBu', or 'viridis'
    :param alpha: transparency (alpha-value) for the std area
    :param show_legend: flag if the legend entry should be printed, set to True when using multiple subplots
    :param title: title displayed above the figure, set to None to suppress the title
    :return: handle to the resulting figure
    """
    if not check_all_types_equal(distributions):
        raise pyrado.TypeErr(msg='Types of all distributions have to be identical!')
    if not isinstance(distributions[0], Distribution):
        raise pyrado.TypeErr(msg='Distributions must be PyTorch Distribution instances!')

    if distr_labels is None:
        distr_labels = [rf'iter\_{i}' for i in range(len(distributions))]

    # Get the color map customized to the number of distributions to plot
    cmap = get_cmap(cmap_name)
    ax.set_prop_cycle(color=cmap(np.linspace(0., 1., max(2, len(distributions)))))

    # Create evaluation grid
    x_gird = to.linspace(x_grid_limits[0], x_grid_limits[1], resolution)

    # Plot the data
    for i, d in enumerate(distributions):
        probs = to.exp(d.log_prob(x_gird))
        ax.plot(x_gird.numpy(), probs.numpy(), label=distr_labels[i])
        ax.fill_between(x_gird.numpy(), np.zeros(probs.size()), probs.numpy(), alpha=alpha)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if show_legend:
        ax.legend(ncol=2)
    if title is not None:
        ax.set_title(title)
    return plt.gcf()
