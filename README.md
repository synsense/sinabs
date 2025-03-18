[![PyPI - Package](https://img.shields.io/pypi/v/sinabs.svg)](https://pypi.org/project/sinabs/)
[![Documentation Status](https://readthedocs.org/projects/sinabs/badge/?version=main)](https://sinabs.readthedocs.io)
[![codecov](https://codecov.io/gh/synsense/sinabs/branch/develop/graph/badge.svg?token=JPGAW4SH1W)](https://codecov.io/gh/synsense/sinabs)
[![PyPI - Downloads](https://img.shields.io/pypi/dd/sinabs)](https://pepy.tech/project/sinabs)
[![Discord](https://img.shields.io/discord/852094154188259338)](https://discord.gg/V6FHBZURkg)
![sinabs](docs/_static/sinabs-logo-lowercase-whitebg.png)

Sinabs (Sinabs Is Not A Brain Simulator) is a python library for the development and implementation of Spiking Convolutional Neural Networks (SCNNs).
The library implements several layers that are `spiking` equivalents of CNN layers.
In addition it provides support to import CNN models implemented in torch conveniently to test their `spiking` equivalent implementation.
This project is managed by SynSense (former aiCTX AG).

The `sinabs-dynapcnn` was incorporated to this project, and it enables porting sinabs models to chips and dev-kits with DYNAP-CNN technology.


Installation
------------
For the stable release on the main branch:
```
pip install sinabs
```
or (thanks to [@Tobias-Fischer](https://github.com/Tobias-Fischer))
```
conda install -c conda-forge sinabs
```

For the latest pre-release on the develop branch that passed the tests:
```
pip install sinabs --pre
```
The package has been tested on the following configurations
[![](http://github-actions.40ants.com/synsense/sinabs/matrix.svg?only=ci.multitest)](https://github.com/synsense/sinabs)


Documentation and Examples
--------------------------
[https://sinabs.readthedocs.io/](https://sinabs.readthedocs.io/)

Questions? Feedback?
--------------------
Please join us on the [#sinabs Discord channel](https://discord.gg/V6FHBZURkg)!

- If you would like to report bugs or push any changes, you can do this on our [github repository](https://github.com/synsense/sinabs/issues).

License
-------
Sinabs is published under Apache v2.0. See the LICENSE file for details.


Contributing to Sinabs
------------------------
Checkout the [contributing](https://sinabs.readthedocs.io/en/develop/about/contributing.html) page for more info.


Citation
--------

In case you find this software library useful for your work please consider citing it as follows:

```
@software{sinabs,
author = {Sheik, Sadique and Lenz, Gregor  and Bauer, Felix and Kuepelioglu, Nogay },
doi = {10.5281/zenodo.8385545},
license = {Apache-2.0},
title = {{SINABS: A simple Pytorch based SNN library specialised for Speck}},
url = {https://github.com/synsense/sinabs}
}
```


