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

import os.path as osp
from typing import Any, Callable

from matplotlib import pyplot as plt


class _LFMEntry:
    """One plot managed by the LiveFigureManager"""

    def __init__(self, update_fcn, title):
        """
        Constructor

        :param update_fcn: function used to update the figures, this actually defines what is to be plotted
        :param title: figure title
        """
        self.update_fcn = update_fcn
        self.title = title
        self._fignum = None

    def update(self, data, args) -> bool:
        """
        Update an individual plot.

        :param data: data to plot
        :param args: parsed command line arguments
        :return
        """
        if self._fignum is None:
            fig = plt.figure(num=self.title)
            self._fignum = fig.number
        elif not plt.fignum_exists(self._fignum):
            # got closed
            return False
        else:
            fig = plt.figure(self._fignum)
        fig.clf()

        # Call drawer
        res = self.update_fcn(fig, data, args)

        # Signal that we're still alive
        if res is False:
            # Cancelled, close figure
            plt.close(fig)
            return False

        return True


class LiveFigureManager:
    """
    Manages multiple matplotlib figures and refreshes them when the input file changes.
    It also ensures that if you close a figure, it does not reappear on the next update.
    If all figures are closed, the update loop is stopped.
    """

    def __init__(self, file_path: str, data_loader: Callable[[str], Any], args, update_interval: int = 3):
        """
        Constructor

        :param file_path: name of file to load updates from
        :param data_loader: called to load the file contents into some internal representation like a pandas `DataFrame`
        :param args: parsed command line arguments
        :param update_interval: time to wait between figure updates [s]
        """
        self._file_path = file_path
        self._data_loader = data_loader
        self._args = args
        self._update_interval = update_interval
        self._figure_list = []

    def figure(self, title: str = None):
        """
        Decorator to define a figure update function.
        Every marked function will be called when the file changes to visualize the updated data.

        :usage:
        .. code-block:: python

            @lfm.figure('A figure')
            def a_figure(fig, data, args):
                ax = fig.add_subplot(111)
                ax.plot(data[...])

        :param title: figure title
        :return: decorator for the plotting function
        """

        def wrapper(func):
            entry = _LFMEntry(func, title)
            self._figure_list.append(entry)
            return entry

        return wrapper

    def _plot_all(self):
        """Load the data and plot all registered figures."""
        data = self._data_loader(self._file_path)
        self._figure_list[:] = [pl for pl in self._figure_list if pl.update(data, self._args)]

    def spin(self):
        """Run the plot update loop."""
        # Create all plots
        plt.ion()
        self._plot_all()

        # Watch modification time
        time_last_plot = osp.getmtime(self._file_path)

        while len(plt.get_fignums()) > 0:
            # Check for changes
            mt = osp.getmtime(self._file_path)

            if mt > time_last_plot:
                # Changed, so update
                self._plot_all()
                time_last_plot = mt

            # Give matplotlib some time
            plt.pause(self._update_interval)
