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

from pyrado import use_pgf


if use_pgf:
    import matplotlib

    matplotlib.use("pgf")
from matplotlib import font_manager
from matplotlib import pyplot as plt


def set_style(style_name: str = "default"):
    """
    Sets colors, fonts, font sizes, bounding boxes, and more for plots using pyplot.

    .. note::
        The font sizes of the predefined styles will be overwritten!

    .. seealso::
        https://matplotlib.org/users/customizing.html
        https://matplotlib.org/tutorials/introductory/customizing.html#matplotlib-rcparams
        https://matplotlib.org/api/_as_gen/matplotlib.pyplot.rc.html
        https://matplotlib.org/users/usetex.html
        https://stackoverflow.com/questions/11367736/matplotlib-consistent-font-using-latex

    :param style_name: str containing the matplotlib style name, or 'default' for the Pyrado default style
    """

    # noinspection PyBroadException
    try:
        font_manager.findSystemFonts(fontpaths=None, fontext="ttf")
        font_manager.findfont("serif", rebuild_if_missing=False)
    except Exception:  # pylint: disable=broad-except
        pass

    if style_name == "default":
        plt.rc("font", family="serif")
        plt.rc("text", usetex=False)
        plt.rc("text.latex", preamble=r"\usepackage{lmodern}")  # direct font input
        plt.rc("mathtext", fontset="cm")
        plt.rc("pgf", rcfonts=False)  # to use the LaTeX document's fonts in the PGF plots
        plt.rc("image", cmap="inferno")  # default: viridis
        plt.rc("legend", frameon=False)
        plt.rc("legend", framealpha=0.4)
        plt.rc("axes", xmargin=0.0)  # disable margins by default
    elif style_name == "ggplot":
        plt.style.use("ggplot")
    elif style_name == "dark_background":
        plt.style.use("dark_background")
    elif style_name == "seaborn":
        plt.style.use("seaborn")
    elif style_name == "seaborn-muted":
        plt.style.use("seaborn-muted")
    else:
        ValueError(
            "Unknown style name! Got {}, but expected 'default', 'ggplot', 'dark_background',"
            "'seaborn', or 'seaborn-muted'.".format(style_name)
        )

    plt.rc("font", size=10)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("savefig", bbox="tight")  # 'tight' is incompatible with pipe-based animation backends
    plt.rc("savefig", pad_inches=0)


set_style()
