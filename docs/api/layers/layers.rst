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

Auxiliary layers
----------------
Cropping2dLayer
~~~~~~~~~~~~~~~
.. autoclass:: Cropping2dLayer
    :members:

Pooling
~~~~~~~
.. autoclass:: SpikingMaxPooling2dLayer
    :members:

SumPool2d
~~~~~~~~~
.. autoclass:: SumPool2d
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

Parent layers
-------------
Other Sinabs layers might inherit from those.

StatefulLayer
~~~~~~~~~~~~~
.. autoclass:: StatefulLayer
    :members:

Reshaping layers
----------------

FlattenTime
~~~~~~~~~~~~~~~~
.. autoclass:: FlattenTime

UnflattenTime
~~~~~~~~~~~~~
.. autoclass:: UnflattenTime


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
