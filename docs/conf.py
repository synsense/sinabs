import sinabs

project = "Sinabs"
copyright = "2019-2022, SynSense"
author = "employees of SynSense"

extensions = [
    "pbr.sphinxext",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST settings
suppress_warnings = ["myst.header"]
# jupyter_execute_notebooks = "off"
execution_timeout = 300

templates_path = ["_templates"]

master_doc = "index"

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

html_title = sinabs.__version__
html_theme = "sphinx_book_theme"
html_logo = "_static/sinabs-logo-lowercase-whitebg.png"
html_show_sourcelink = True
html_sourcelink_suffix = ""

html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://github.com/synsense/sinabs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "repository_branch": "develop",
    "path_to_docs": "docs",
    "use_fullscreen_button": True,
}
