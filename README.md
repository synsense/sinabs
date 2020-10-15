![PyPI - Package](https://img.shields.io/pypi/v/sinabs.svg) [![Documentation Status](https://img.shields.io/badge/docs-ok-green)](https://aictx.gitlab.io/sinabs) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sinabs?logo=python) ![Black - formatter](https://img.shields.io/badge/code%20style-black-black) ![PyPI - Downloads](https://img.shields.io/pypi/dd/sinabs)


SINABS
======

Getting started
---------------

**Sinabs Is Not A Brain Simulator**

`sinabs` is a python library for development and implementation of Spiking Convolutional Neural Networks (SCNNs).
The library implements several layers that are `spiking` equivalents of CNN layers.
In addition it provides support to import CNN models implemented in torch conveniently to test their `spiking` equivalent implementation.
This project is managed by SynSense (former aiCTX AG).

**NOTE**: The conversion of CNNs to SCNNs is still a subject of research and we strive to keep the library updated to the state-of-the art in addition to providing options to compare various approaches both at a high level abstraction to low level implementation details.

**NOTE**: This library is in Beta release stage and is subject to API changes.

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
$ git clone https://gitlab.com/aiCTX/sinabs.git>
$ cd sinabs
$ pip install -e . --user
```

For developers, we recommend that you install this package as a development version so that you can update the package without reinstalling the package.


Documentation and Examples
--------------------------

[https://sinabs.ai](https://sinabs.ai)


If you would like to generate documentation locally, you can do that using `sphinx`.

**REQUIREMENT** You will require `pandoc` installed on your system.

You can generate a sphinx documentation for this package by running the the following command.

```
$ cd /path/to/sinabs/
$ pip install -r sphinx-requirements.txt
$ python setup.py build_sphinx
```

This will build and auto generate html documentation at `docs/build/html/index.html`
You can access the generated documentation in your browser.
```
$ firefox docs/build/html/index.html
```

License
-------

`sinabs` is published under AGPL v3.0. See the LICENSE file for details.


Contributing to `sinabs`
------------------------

Checkout [CONTRIBUTING.md](https://sinabs.ai/contributing.html)
