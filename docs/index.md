<img src="_static/sinabs-logo-lowercase.png" alt="logo" width="500"/>

![PyPI - Package](https://img.shields.io/pypi/v/sinabs.svg) 
[![Documentation Status](https://img.shields.io/badge/docs-ok-green)](https://aictx.gitlab.io/sinabs) 
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sinabs?logo=python) 
![Black - formatter](https://img.shields.io/badge/code%20style-black-black) 
![PyPI - Downloads](https://img.shields.io/pypi/dd/sinabs)

### **Sinabs Is Not A Brain Simulator**
It's a deep learning library based on PyTorch for spiking neural networks, with a focus on simplicity, fast training and extendability. Sinabs works well for Vision models because of its support for weight transfer. If you're looking to work with audio data or different backends, be sure to check out [Rockpool](https://rockpool.ai/) as well.

### Getting started
* **{doc}`Install Sinabs<getting_started/install>`** and potentially some plugins.
* **{doc}`Dive right into the syntax<getting_started/quickstart>`** if you know your way around SNN simulators.

### Tutorials
* **{doc}`Convert an existin ANN<tutorials/weight_transfer_mnist>`** if you want to get started quickly.
* **{doc}`Run a first example using BPTT<tutorials/bptt>`** with this neuromorphic version of the MNIST dataset.

### Plugins
* Deploying models to neuromorphic hardware: [Sinabs-DynapCNN](https://synsense.gitlab.io/sinabs-dynapcnn/).
* Training feed-forward models 10x faster: [EXODUS](https://github.com/synsense/sinabs-exodus).

### API reference
* **{doc}`Complete reference overview<api/api>`**. 
* **{doc}`Supported neuron models<api/layers>`**.
* **{doc}`Weight transfer API<api/from_torch>`**. 

### About
* **{doc}`About Sinabs<about/about>`**. How the project came about.
* **{doc}`Contribution guidelines<about/contributing>`**. Please read this before opening a pull request.
* **{doc}`Release notes<about/release_notes>`**. Version changes.

### Contact
* **{doc}`Contact us<contact>`**. For questions and bug reports!

```{toctree}
:hidden:
getting_started/getting_started
auto_examples/index
tutorials/tutorials
how_tos/how_tos
plugins/plugins
api/api
about/about
contact
```
