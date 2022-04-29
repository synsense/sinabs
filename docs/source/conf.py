# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sinabs.backend.dynapcnn

# -- Project information -----------------------------------------------------

project = 'sinabs-dynapcnn'
copyright = '2020-2022, Synsense AG'
author = 'Sadique Sheik, Martino Sorbaro, Felix Bauer'

# The full version, including alpha/beta/rc tags
# release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    #'nbsphinx',
    "pbr.sphinxext",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.graphviz',
    'sphinx.ext.mathjax',
    "myst_nb",
    #'m2r2',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST settings
suppress_warnings = ["myst.header"]
jupyter_execute_notebooks = "off"

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

#source_suffix = {".rst": 'restructuredtext',
#                 ".txt": 'markdown',
#                 ".md": 'markdown',
#                 }

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_title = sinabs.backend.dynapcnn.__version__
html_theme = "sphinx_book_theme"
html_logo = "_static/sinabs-logo-lowercase-whitebg.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_theme_options = {
    "repository_url": "https://gitlab.com/synsense/sinabs-dynapcnn",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_fullscreen_button": True,
}



# Include __init__ docstring in method documentation
autoclass_content = 'both'

# Include return type in line
napoleon_use_rtype = False

# API module name display
add_module_names = False
