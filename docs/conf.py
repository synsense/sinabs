import sinabs

project = "Sinabs"
copyright = "2019-present, SynSense"
author = "employees of SynSense"

master_doc = "index"

extensions = [
    "myst_nb",
    "pbr.sphinxext",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": "gallery/",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    # "backreferences_dir": "gen_modules/backreferences",
    "doc_module": ("sinabs",),
    "download_all_examples": False,
    "ignore_pattern": r"utils\.py",
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# MyST settings
suppress_warnings = ["myst.header"]
nb_execution_timeout = 300
nb_execution_excludepatterns = [
    "LeNet_5_EngChinese.ipynb",
    "bptt.ipynb",
    "weight_transfer_mnist.ipynb",
]
# nb_execution_mode = "off"

templates_path = ["_templates"]

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
