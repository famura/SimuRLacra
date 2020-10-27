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

"""
Functions to plot Pyrado policies
"""
import numpy as np
import torch.nn as nn
from matplotlib import ticker, colorbar
from matplotlib import pyplot as plt
from typing import Any

import pyrado
from pyrado.plotting.utils import AccNorm
from pyrado.policies.adn import ADNPolicy
from pyrado.policies.base import Policy
from pyrado.policies.neural_fields import NFPolicy
from pyrado.utils.data_types import EnvSpec
from pyrado.utils.input_output import print_cbt


def _annotate_img(img,
                  data: [list, np.ndarray] = None,
                  thold_lo: float = None,
                  thold_up: float = None,
                  valfmt: str = '{x:.2f}',
                  textcolors: tuple = ('white', 'black'),
                  **textkw: Any):
    """
    Annotate a given image.

    .. note::
        The text color changes based on thresholds which only make sense for symmetric color maps.

    :param mg: AxesImage to be labeled.
    :param data: data used to annotate. If None, the image's data is used.
    :param thold_lo: lower threshold for changing the color
    :param thold_up: upper threshold for changing the color
    :param valfmt: format of the annotations inside the heat map. This should either use the string format method, e.g.
                   '$ {x:.2f}', or be a :class:matplotlib.ticker.Formatter.
    :param textcolors: two color specifications. The first is used for values below a threshold,
                       the second for those above.
    :param textkw: further arguments passed on to the created text labels
    """
    if not isinstance(data, (list, np.ndarray)):
        data = img.get_array()

    # Normalize the threshold to the images color range
    if thold_lo is None:
        thold_lo = data.min()*0.5
    if thold_up is None:
        thold_up = data.max()*0.5

    # Set default alignment to center, but allow it to be overwritten by textkw
    kw = dict(horizontalalignment='center', verticalalignment='center')
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a text for each 'pixel'.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[thold_lo < data[i, j] < thold_up])  # if true then use second color
            text = img.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)


def render_policy_params(policy: Policy,
                         env_spec: EnvSpec,
                         cmap_name: str = 'RdBu',
                         ax_hm: plt.Axes = None,
                         annotate: bool = True,
                         annotation_valfmt: str = '{x:.2f}',
                         colorbar_label: str = '',
                         xlabel: str = None,
                         ylabel: str = None,
                         ) -> plt.Figure:
    """
    Plot the weights and biases as images, and a color bar.

    .. note::
        If you want to have a tight layout, it is best to pass axes of a figure with `tight_layout=True` or
        `constrained_layout=True`.

    :param policy: policy to visualize
    :param env_spec: environment specification
    :param cmap_name: name of the color map, e.g. 'inferno', 'RdBu', or 'viridis'
    :param ax_hm: axis to draw the heat map onto, if equal to None a new figure is opened
    :param annotate: select if the heat map should be annotated
    :param annotation_valfmt: format of the annotations inside the heat map, irrelevant if annotate = False
    :param colorbar_label: label for the color bar
    :param xlabel: label for the x axis
    :param ylabel: label for the y axis
    :return: handles to figures
    """
    if not isinstance(policy, nn.Module):
        raise pyrado.TypeErr(given=policy, expected_type=nn.Module)
    cmap = plt.get_cmap(cmap_name)

    # Create axes and subplots depending on the NN structure
    num_rows = len(list(policy.parameters()))
    fig = plt.figure(figsize=(14, 10), tight_layout=False)
    gs = fig.add_gridspec(num_rows, 2, width_ratios=[14, 1])  # right column is the color bar
    ax_cb = fig.add_subplot(gs[:, 1])

    # Accumulative norm for the colors
    norm = AccNorm()

    for i, (name, param) in enumerate(policy.named_parameters()):
        # Create current axis
        ax = plt.subplot(gs[i, 0])
        ax.set_title(name.replace('_', r'\_'))

        # Convert the data and plot the image with the colors proportional to the parameters
        if param.ndim == 3:
            # For example convolution layers
            param = param.flatten(0)
            print_cbt(f'Flattened the first dimension of the {name} parameter tensor.', 'y')
        data = np.atleast_2d(param.detach().cpu().numpy())

        img = plt.imshow(data, cmap=cmap, norm=norm, aspect='auto', origin='lower')

        if annotate:
            _annotate_img(
                img,
                thold_lo=0.75*min(policy.param_values).detach().cpu().numpy(),
                thold_up=0.75*max(policy.param_values).detach().cpu().numpy(),
                valfmt=annotation_valfmt
            )

        # Prepare the ticks
        if isinstance(policy, ADNPolicy):
            if name == 'obs_layer.weight':
                ax.set_xticks(np.arange(env_spec.obs_space.flat_dim))
                ax.set_yticks(np.arange(env_spec.act_space.flat_dim))
                ax.set_xticklabels(env_spec.obs_space.labels)
                ax.set_yticklabels(env_spec.act_space.labels)
            elif name in ['obs_layer.bias', 'nonlin_layer.log_weight', 'nonlin_layer.bias']:
                ax.set_xticks(np.arange(env_spec.act_space.flat_dim))
                ax.set_xticklabels(env_spec.act_space.labels)
                ax.yaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            elif name == 'prev_act_layer.weight':
                ax.set_xticks(np.arange(env_spec.act_space.flat_dim))
                ax.set_yticks(np.arange(env_spec.act_space.flat_dim))
                ax.set_xticklabels(env_spec.act_space.labels)
                ax.set_yticklabels(env_spec.act_space.labels)
            elif name in ['_log_tau', '_log_kappa', '_log_capacity']:
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            else:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        elif isinstance(policy, NFPolicy):
            if name == 'obs_layer.weight':
                ax.set_xticks(np.arange(env_spec.obs_space.flat_dim))
                ax.yaxis.set_major_locator(ticker.NullLocator())
                ax.set_xticklabels(env_spec.obs_space.labels)
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            elif name in ['_log_tau', '_log_kappa', '_potentials_init', 'resting_level', 'obs_layer.bias',
                          'conv_layer.weight', 'nonlin_layer.log_weight', 'nonlin_layer.bias']:
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.yaxis.set_major_locator(ticker.NullLocator())
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            elif name == 'act_layer.weight':
                ax.xaxis.set_major_locator(ticker.NullLocator())
                ax.set_yticks(np.arange(env_spec.act_space.flat_dim))
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
                ax.set_yticklabels(env_spec.act_space.labels)
            else:
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

        # Add the color bar (call this within the loop to make the AccNorm scan every image)
        colorbar.ColorbarBase(ax_cb, cmap=cmap, norm=norm, label=colorbar_label)

    # Increase the vertical white spaces between the subplots
    plt.subplots_adjust(hspace=.7, wspace=0.1)

    # Set the labels
    if xlabel is not None:
        ax_hm.set_xlabel(xlabel)
    if ylabel is not None:
        ax_hm.set_ylabel(ylabel)

    return fig
