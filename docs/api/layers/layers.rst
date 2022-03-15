layers
======

.. currentmodule:: sinabs.layers


All the layers implemented in this package can be used similar to `torch.nn` layers in your implementations.

Main spiking layers
-------------------
Those layers make use of the sinabs.activation module. 

.. toctree::
    iaf
    lif
    alif

Non-spiking layers
------------------
.. toctree::
    exp_leak

Pooling
-------
.. autoclass:: SpikingMaxPooling2dLayer
    :members:

.. autoclass:: SumPool2d
    :members:

Conversion from images / analog signals
---------------------------------------
The hybrid layers have inputs and outputs of different formats (eg. take analog values as inputs and produce spikes as outputs.)

.. autoclass:: Img2SpikeLayer
    :members:

.. autoclass:: Sig2SpikeLayer
    :members:

Parent layers
-------------
Other Sinabs layers might inherit from those.

.. autoclass:: StatefulLayer
    :members:

.. autoclass:: SqueezeMixin
    :members:

Auxiliary layers
----------------
.. autoclass:: Cropping2dLayer
    :members:

.. autoclass:: FlattenTime
    :members:

.. autoclass:: UnflattenTime
    :members:

ANN layers
----------
These are utility layers used in the training of ANNs, in order to provide specific features suitable for SNN conversion.

.. autoclass:: NeuromorphicReLU
    :members:

.. autoclass:: QuantizeLayer
    :members:
