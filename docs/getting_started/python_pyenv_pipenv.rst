Python versions, pyenv and pipenv
*********************************

Author: Sadique Sheik

Getting your python environment up and running could be complicated and cumbersome when dealing with multiple packages and corresponding dependencies.
Even more so when your are working with an operating system like `Arch Linux` which is always on the bleeding edge but your python packages do not yet support the latest and greatest version of python installed on your system.

There are two ways to deal with such a problem (other than deciding to never upgrading your operating system, which would be a terrible idea!!)

    1. Use Anaconda

    2. Work with pyenv (and a virtual environment which does not interfere with your system python or the various paths)

If you are an Anaconda user, then this short tutorial is not for you. However, you can simply install `sinabs` using `conda install sinabs -c conda-forge`. All the best!

Here I will describe to you my preferred approach to dealing with python versions and dependencies: **pyenv + pipenv**

Introduction
============

The two part solution to these problems is the following.

**pyenv**: To choose your version of python
-------------------------------------------

`pyenv` is a tool that helps you install and maintain multiple versions of python in parallel as a system user (without needing to be a system admin/root)

For instance if you needed to install python version 3.7.5 for your project, you would run the following command::

    $ pyenv install 3.7.5

This install python 3.7.5 in your home directory at ``~/.pyenv/versions/3.7.5/``.
Note that this python is installed in a secluded location and does not interfere with your system python or other paths.

**pipenv**: To maintain your virtual environment
------------------------------------------------

Okay, now what?
Now that we have our preferred python installed, we want to be able to use our project with this installation.
`pipenv` allows us to create virtual environments with the specified version of python and maintain all the necessary dependencies for your project.

You navigate to your project directory and do the follwing::

    $ pipenv --python ~/.pyenv/versions/3.7.5/bin/python

.. Note::

    If the pyenv directory is in your path, you can simply invoke ``$ pipenv --python 3.7`` without you having to explicitly specify the path.

This will crate a virtual environment with the corresponding python version and a `Pipfile` in your project directory.
From the top level of your project directory, you can invoke this virtual environment by calling::

    $ pipenv shell

Once you are in, you should be able to install any packages with `pip` within this virtual environment.
For software `packages` it is perhaps better to install dependencies using pipenv to add them to the project `Pipfile` which could then be distributed with your package.
The `Pipfile` essentially replace `requirements.txt` for `pipenv`.

You can checkout the `Pipfile` of `sinabs` to get a feel for it.

PEW: Quick access
=================

You would have noticed that to activate your shell you need to navigate to your project directory.
This might get a bit tiresome if you have your project hidden in a deep directory structure, or if you just don't know where it is located on this machine.

In order to activate your virtual environment from an arbitrary file path in your terminal, you can use a tool called `pew`.
To list all available virtual environments, run::

    $ pew ls

This should list a set of virtual environments available on your system. Let's say your project's env. was called `MySinabsProject-sdfgaa`.
You can activate your project's virtual environment as::

    $ pew workon MySinabsProject-sdfgaa

This will activate a shell for your project and take you to the project directory.

.. Tip::

    You might have noticed that there is no indication of the current virtual environment when you use `pew` as we did above.
    This can be remedied by adding the following line to your .bashrc/.zshrc file: `source $(pew shell_config)` .

Checkout the links in the references section below for detailed explanations and tutorials.

TL;DR : Quick steps for a project using `sinabs`
================================================

Make sure you have `pipenv` and `pyenv` installed on your system.

Command line:
-------------

Launch a terminal and fire away::

    $ cd <to/your/project/folder>
    $ pyenv install 3.7.5
    $ pipenv --python ~/.pyenv/versions/3.7.5/bin/python
    $ pipenv shell
    (project-venv)$ pip install sinabs
    (project-venv)$ python your_script_with_sinabs.py

You can obviously install ipython, jupyter-notebook etc within this venv.

Pycharm IDE:
------------
Pipenv is also supported by `pycharm`. You can can set your `pycharm project interpreter to use pipenv <https://www.jetbrains.com/help/pycharm/pipenv.html>`_.


References
==========

- `Pipenv: A Guide to the New Python Packaging Tool <https://realpython.com/pipenv-guide/>`_
- `Installing Python packages in 2019: pyenv and pipenv <https://gioele.io/pyenv-pipenv>`_
- `Basic Usage of Pipenv <https://pipenv-fork.readthedocs.io/en/latest/basics.html>`_
- `Pycharm: Configuring a Pipenv environment. <https://www.jetbrains.com/help/pycharm/pipenv.html>`_
