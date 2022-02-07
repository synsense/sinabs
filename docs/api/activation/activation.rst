activation
==========

.. currentmodule:: sinabs.activation

spiking activation
------------------

The main object to pass to a spiking layer is an ActivationFunction, which combines spike generation, reset mechanism and surrogate gradient function.

.. toctree::
   :maxdepth: 1

   spike_generation
   reset_mechanism
   surrogate_gradient

ActivationFunction
^^^^^^^^^^^^^^^^^^
.. autoclass:: ActivationFunction

ALIFActivationFunction
^^^^^^^^^^^^^^^^^^^^^^
.. autoclass:: ALIFActivationFunction


quantization activation
-----------------------

.. automodule:: sinabs.activation.quantize
   :members:
   :undoc-members:
