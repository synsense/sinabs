Fundamentals of Sinabs
======================

Sinabs: PyTorch Extension
-------------------------

Sinabs is designed to work along side pytorch by providing additional functionality to support spiking neural network dynamics.
In practice, Sinabs comprises a set of activation `layers` that emulate spiking neuronal dynamics.
Because of the temporal nature of SNNs, several layers in sinabs are `stateful`.

Sinabs layers augment the activation functions in pytorch, and are designed to work along side the various connection layers in pytorch such as `AvgPool2d`, `Linear` and `Conv2d` layers.

Data convention: float tensors
------------------------------

Although Sinabs deals with spiking data or events, for considerations of simulation speed and learning efficiency, all the layers are designed to work with spike rasters or tensors.
In addition, for compatibility with other layers in pytorch, the input data needs to be a float tensor (even if the original data happens to be binary).

Sinabs respects pytorch's data covention where ever appropriate.
Because Sinabs deals with temporal data most often, there is an additional time axis expected in the input data.

So, the data convention is:

.. code-block:: python

    (Batch, Time, ...)


For instance, if you are working with data from a 2d sensor such as the DVS, your data's dimensions would be ordered as follows:

.. code-block:: python

    (Batch, Time, Polarity, Height, Width)

The activity of a population of neuron (arranged linearly) would be represented as:

.. code-block:: python

    (Batch, Time, NeuronId)


Squeeze layers: 4D layers and 5D tensors
----------------------------------------

If you are working with Conv2d and other layers in pytorch that work with 2D/image data, you come across a problem with spiking data.
These layers only understand 4D tensors (Batch, Channel, Height, Width). They have no means of supporting the additional time axis we have in our spiking data.

One way to address this issue is by reducing the 5D data to 4D, then passing it through a 4D layer and then reshaping the output back to 5D.
This approach of continually reshaping data back and forth potentially make the model a bit more cumbersome. So Sinabs provides variants of the original spiking layers.

- Squeeze layers are initialized with a given batch size.
- The expect the first dimension in the data to comprise of both batch and time.
- Their output is also of the same form as their input.

What this means is, these layers expect your data to be of the form

.. code-block:: python

    (Batch*Time, ...) # For generic data
    (Batch*Time, Channel, Height, Width)  # For 2D sensor data like DVS


Getting SNNs to work
--------------------

There are two main approaches in Sinabs to building functional SNNs.

#. :doc:`Conversion/Weight transfer from ANNs <../tutorials/weight_transfer_mnist>`
#. :doc:`Directly training SNNs with BPTT <../tutorials/bptt>`

You will find examples on both these approaches in the tutorials section of this documentation.