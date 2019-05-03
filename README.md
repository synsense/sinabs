SINABS
======

Getting started
---------------

**Sinabs Is Not A Brain Simulator**

**NOTE**: This project is mirrored to gitlab.com/ai-ctx/sinabs and is managed by aiCTX AG.

`sinabs` (pytorch based library) is developed to design and implement Spiking Convolutional Neural Networks (SCNNs).
The library implements several layers that are `spiking` equivalents of CNN layers.
In addition it provides support to import CNN models implemented in keras conveniently to test their `spiking` equivalent implementation.

**NOTE**: The conversion of CNNs to SCNNs is still a subject of research and we strive to keep the library updated to the state-of-the art in addition to providing options to compare various approaches both at a high level abstraction to low level implementation details.

**NOTE**: This library is an alpha versions and is subject to API changes.

Installation
------------

Download the package from git:

```
$ git clone <URLto/sinabs.git>
```

*NOTE*: Replace `<URLto/sinabs.git>` with appropriate git url.

We recommend that you install this package as a development version so that you can update the package without reinstalling the package

```
$ cd path/to/sinabs
$ pip install -e . --user
```

Documentation
-------------

You can generate a sphynx documentation for this package by running the the following command.

*Requirements*: sphinx, pandoc, nbsphinx

```
$ cd /path/to/sinabs/docs/
$ make html
$ firefox build/html/index.html
```

This will build and auto generate html documentation at `sinabs/docs/build/html/index.html`

License
-------

`sinabs` is published under AGPL v3.0. See the LICENSE file for details.


Contributing to `sinabs`
------------------------

Checkout [CONTRIBUTING.md](CONTRIBUTING.md)
