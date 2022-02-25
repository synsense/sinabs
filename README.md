![PyPI - Package](https://img.shields.io/pypi/v/sinabs.svg) 
[![Documentation Status](https://readthedocs.org/projects/sinabs/badge/?version=latest)](https://sinabs.readthedocs.io/en/latest/?badge=latest)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sinabs?logo=python) 
![Black - formatter](https://img.shields.io/badge/code%20style-black-black) 
![PyPI - Downloads](https://img.shields.io/pypi/dd/sinabs)
[![](http://github-actions.40ants.com/synsense/sinabs/matrix.svg?only=ci.multitest.ubuntu-latest)](https://github.com/synsense/sinabs)

![sinabs](docs/_static/sinabs-logo-lowercase.png)

Getting started
---------------
`sinabs` (Sinabs Is Not A Brain Simulator) is a python library for development and implementation of Spiking Convolutional Neural Networks (SCNNs).
The library implements several layers that are `spiking` equivalents of CNN layers.
In addition it provides support to import CNN models implemented in torch conveniently to test their `spiking` equivalent implementation.
This project is managed by SynSense (former aiCTX AG).

Installation
------------

You can install `sinabs` with pip:

```
pip install sinabs
```
Checkout our quick instructional on how to create a project based on `sinabs` within a virtual environment using [pyenv+pipenv](https://sinabs.ai/howto/python_pyenv_pipenv.html)

If you want to develop or have access to source code of `sinabs`, download the package from the git repository:

```
$ cd <to/your/software/folder>
$ git clone git@github.com:synsense/sinabs.git
$ cd sinabs
$ pip install -e . --user
```

For developers, we recommend that you install this package as a development version so that you can update the package without reinstalling the package.


Documentation and Examples
--------------------------

[https://sinabs.ai](https://sinabs.ai)


If you would like to generate documentation locally, you can do that using `sphinx`.

You can generate a sphinx documentation for this package by running the following command.

```
$ cd /path/to/sinabs/docs
$ pip install -r requirements.txt
$ make html
```

This will build and auto generate html documentation at `docs/_build/html/index.html`
You can access the generated documentation in your browser.
```
$ firefox docs/_build/html/index.html
```

License
-------

`sinabs` is published under AGPL v3.0. See the LICENSE file for details.


Contributing to `sinabs`
------------------------

Checkout [CONTRIBUTING.md](https://sinabs.ai/contributing.html)
