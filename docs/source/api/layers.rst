sinabs.layers
=============

.. toctree::
    :maxdepth: 3
    :caption: sinabs.layers


.. automodule:: sinabs.layers
.. currentmodule:: sinabs.layers


All the layers implemented in this package can be used similar to `torch.nn` layers in your implementations.


Main spiking layer
------------------


`SpikingLayer`
~~~~~~~~~~~~~~

.. autoclass:: SpikingLayer
    :members:


Auxiliary spiking layers
------------------------


`Cropping2dLayer`
~~~~~~~~~~~~~~~~~

.. autoclass:: Cropping2dLayer
    :members:


`SpikingMaxPooling2dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpikingMaxPooling2dLayer
    :members:

`InputLayer`
~~~~~~~~~~~~

.. autoclass:: InputLayer
    :members:


Hybrid layers
-------------

The hybrid layers have inputs and outputs of different formats (eg. take analog values as inputs and produce spikes as outputs.)

`Img2SpikeLayer`
~~~~~~~~~~~~~~~~

.. autoclass:: Img2SpikeLayer
    :members:

`Sig2SpikeLayer`
~~~~~~~~~~~~~~~~

.. autoclass:: Sig2SpikeLayer
    :members:

ANN layers
----------

These are utility layers used in the training of ANNs, in order to provide specific features suitable for SNN conversion.

`NeuromorphicReLU`
~~~~~~~~~~~~~~~~~~

.. autoclass:: NeuromorphicReLU
    :members:

`QuantizeLayer`
~~~~~~~~~~~~~~~

.. autoclass:: QuantizeLayer
    :members:

`SumPool2d`
~~~~~~~~~~~

.. autoclass:: SumPool2d
    :members:


sinabs.layers.functional
========================


.. automodule:: sinabs.layers.functional.threshold
  :members:

.. automodule:: sinabs.layers.functional.quant
  :members:
