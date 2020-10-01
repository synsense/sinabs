sinabs.layers
===============

.. automodule:: sinabs.layers
.. currentmodule:: sinabs.layers

Layers
------

All the layers implemented in this package can be used similar to `torch.nn` layers in your implementations.
The layer initialization expects an input shape which follows the below format::

    inputShape : (channels, height, width)  # For 2d Layers
    inputShape : (length, )  # For flat layers

.. note:: The input shape does not include the dimension of batch size.

Abstract layers
---------------


`SpikingLayer`
~~~~~~~~~~~~~~

.. autoclass:: SpikingLayer
    :members:


Full implementations
--------------------


`Cropping2dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Cropping2dLayer
    :members:

`FlattenLayer`
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: FlattenLayer
    :members:

`SpikingConv1dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpikingConv1dLayer
    :members:

`SpikingConv2dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpikingConv2dLayer
    :members:

`SpikingConv3dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpikingConv3dLayer
    :members:

`SpikingMaxPooling2dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SpikingMaxPooling2dLayer
    :members:

`InputLayer`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: InputLayer
    :members:

`QuantizeLayer`
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: QuantizeLayer
    :members:

`Sumpooling2dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SumPooling2dLayer
    :members:

`ZeroPad2dLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: ZeroPad2dLayer
    :members:

Hybrid layers
-------------

The hybrid layers have inputs and outputs of different formats (eg. take analog values as inputs and produce spikes as outputs.)

`Img2SpikeLayer`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Img2SpikeLayer
    :members:

`YOLOLayer`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: YOLOLayer
    :members:

ANN layers
-----------------

`NeuromorphicReLU`
~~~~~~~~~~~~~~~~~~

.. autoclass:: NeuromorphicReLU
    :members:

`SumPool2d`
~~~~~~~~~~~

.. autoclass:: SumPool2d
    :members:
