# -*- coding: utf-8 -*-

#  Copyright (c) 2019-2020 SynSense.
#
#  This file is part of sinabs
#
#  sinabs is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sinabs is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sinabs.  If not, see <https://www.gnu.org/licenses/>.

#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# import sinabs
# sys.path.append(os.path.join(os.path.dirname(__name__), ".."))


# -- Project information -----------------------------------------------------

project = "Sinabs"
copyright = "2019-2022, SynSense"
author = "employees of SynSense"

# The short X.Y version
# version = "0.1"
# The full version, including alpha/beta/rc tags
# release = ""

extensions = [
    # "pbr.sphinxext",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb"
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST settings
suppress_warnings = ["myst.header"]
# jupyter_execute_notebooks = "off"

templates_path = ["_templates"]

master_doc = "index"

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_theme = "sphinx_book_theme"
html_logo = "_static/sinabs-logo-lowercase.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_static_path = ["_static"]

html_theme_options = {
    "logo_only": True,
    "repository_url": "https://sinabs.ai",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_fullscreen_button": True,
}
