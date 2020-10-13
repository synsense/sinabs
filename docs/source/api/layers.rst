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

Quantization tools
------------------

`quantize`
~~~~~~~~~~

.. function:: quantize(x)

   PyTorch-compatible function that applies a floor() operation on the input,
   while providing a surrogate gradient (equivalent to that of a linear
   function) in the backward pass.

`stochastic_rounding`
~~~~~~~~~~~~~~~~~~~~~

.. function:: stochastic_rounding(x)

   PyTorch-compatible function that applies stochastic rounding. The input x
   is quantized to ceil(x) with probability (x - floor(x)), and to floor(x)
   otherwise. The backward pass is provided as a surrogate gradient
   (equivalent to that of a linear function).

Thresholding tools
------------------

`threshold_subtract`
~~~~~~~~~~~~~~~~~~~~

.. function:: threshold_subtract(data, threshold=1, window=0.5)

   PyTorch-compatible function that returns the number of spikes emitted,
   given a membrane potential value and in a "threshold subtracting" regime.
   In other words, the integer division of the input by the threshold is returned.
   In the backward pass, the gradient is zero if the membrane is at least
   `threshold - window`, and is passed through otherwise.

`threshold_reset`
~~~~~~~~~~~~~~~~~

.. function:: threshold_reset(data, threshold=1, window=0.5)

   Same as `threshold_subtract`, except that the potential is reset, rather than
   subtracted. In other words, only one output spike is possible.
