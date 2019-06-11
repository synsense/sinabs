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

`TorchLayer`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: TorchLayer
    :members:

`SpikingLayer`
~~~~~~~~~~~~~~

.. autoclass:: SpikingLayer
    :members:


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

`Img2SpikeLayer`
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: Img2SpikeLayer
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

