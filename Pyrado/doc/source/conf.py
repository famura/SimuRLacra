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

# Configuration file for the Sphinx documentation builder.
# This file only contains a selection of the most common options. For a full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# ----------
# Path setup
# ----------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from pyrado import VERSION


# Add packages to the path
sys.path.insert(0, os.path.abspath(".."))  # doc folder
sys.path.insert(0, os.path.abspath("../.."))  # Pyrado folder

# -------------------
# Project information
# -------------------
project = "Pyrado"
version = ".".join(VERSION.split("."))  # short version
release = VERSION  # full version including tags
# noinspection PyShadowingBuiltins
copyright = "2020"  # pylint: disable=redefined-builtin
author = "Fabio Muratore & Felix Treede & Robin Menzenbach & Chrisitan Eilers"

# ---------------------
# General configuration
# ---------------------
# Add any Sphinx extension module names here, as strings
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones
extensions = [
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.autosectionlabel",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
]

# Add any paths that contain templates here, relative to this directory
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and dirs to ignore when looking for source files
# This pattern also affects html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use
pygments_style = "sphinx"

source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -----------------------
# Options for HTML output
# -----------------------
# The theme to use for HTML and HTML Help pages. See the documentation for a list of builtin themes
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here, relative to this directory.
# They are copied after the builtin static files, so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

add_module_names = False
