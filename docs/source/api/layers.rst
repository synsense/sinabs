sinabs.layers
=============

.. toctree::
    :maxdepth: 3
    :caption: sinabs.layers


.. automodule:: sinabs.layers
.. currentmodule:: sinabs.layers


All the layers implemented in this package can be used similar to `torch.nn` layers in your implementations.

Main spiking layers
-------------------

SpikingLayer
~~~~~~~~~~~~
.. autoclass:: SpikingLayer
    :members:

Leaky Integrate Fire (LIF) layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: LIF
    :members:

Adaptive Leaky Integrate Fire (ALIF) layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ALIF
    :members:

Auxiliary spiking layers
------------------------
Cropping2dLayer
~~~~~~~~~~~~~~~
.. autoclass:: Cropping2dLayer
    :members:

SpikingMaxPooling2dLayer
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: SpikingMaxPooling2dLayer
    :members:

InputLayer
~~~~~~~~~~
.. autoclass:: InputLayer
    :members:

Hybrid layers
-------------
The hybrid layers have inputs and outputs of different formats (eg. take analog values as inputs and produce spikes as outputs.)

Img2SpikeLayer
~~~~~~~~~~~~~~
.. autoclass:: Img2SpikeLayer
    :members:

Sig2SpikeLayer
~~~~~~~~~~~~~~
.. autoclass:: Sig2SpikeLayer
    :members:

ANN layers
----------
These are utility layers used in the training of ANNs, in order to provide specific features suitable for SNN conversion.

NeuromorphicReLU
~~~~~~~~~~~~~~~~
.. autoclass:: NeuromorphicReLU
    :members:

QuantizeLayer
~~~~~~~~~~~~~
.. autoclass:: QuantizeLayer
    :members:

SumPool2d
~~~~~~~~~
.. autoclass:: SumPool2d
    :members:


sinabs.layers.functional
========================
Quantization tools
------------------
.. automodule:: sinabs.layers.functional.quantize
   :members:
   :undoc-members:

Thresholding tools
------------------
.. automodule:: sinabs.layers.functional.threshold
   :members:
   :undoc-members:
   