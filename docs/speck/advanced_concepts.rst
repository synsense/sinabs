Advanced
========

Under the hood
--------------

The DynapcnnNetwork converts a given model into a sequence of DVSLayer (at most 1) and DynapcnnLayers.

.. graphviz::

    digraph {
        subgraph cluster {
            node [shape=polygon, sides=4]
            label = "DynapcnnNetwork";
            DVSLayer -> "DynapcnnLayer[0]" -> "DynapcnnLayer[1]" -> "...";
        }
    }




A `ConfigBuilder` then converts this model to a config object when `make_config` method is called. 
`make_config` is called internally when a user calls the `to` method.

.. graphviz::

    digraph {
        node [shape=polygon, sides=4]
        rankdir=LR
        subgraph cluster {
            label = "ConfigBuilder"
            DynapcnnNetwork -> "Samna Config"
        }
    }

`ChipFactory` is used to fetch the appropriate `ConfigBuilder` for a given `device`.

.. graphviz::

    digraph {
        node [shape=polygon, sides=4]
        ChipFactory -> ConfigBuilder
    }


The config object
-----------------

The config object is a nested data structure in samna.
Each device type has its own structure of the Config object.

Memory constraints
------------------

Each spiking CNN core (`DynapcnnLayer`) comprises three memory blocks:

    - Kernel: To store the weights of the convolution
    - Bias: To store the biases
    - Neuron: To store the neuron states

The physical devices have a limited amount of memory available for each of its CNN cores/layers.
Depending on the device architecture, each core could have a different amount of memory available.
When a model is deployed on a device, one needs to ensure that each of the layers has the required amount of momory available to it.
This is take care of by a mapping algorithm when `chip_layers_ordering="auto"` option is set while calling `make_config()`.



Attributes of interest
----------------------

Knowing the mapping of the various layers of the model to the layers of the chip is crucial.
`DynapcnnNetwork.chip_layers_ordering` is a list of chip layer indices where a model was mapped.
This is useful when generating or interpreting events from `samna`, where the `layer` attribute refers to the layer on the chip.

It is important to note here that the `chip_layers_ordering` is only pertinent to `DynapcnnLayer` and does not include the `DVSLayer`.
This is because there is no ambiguity as to where the `DVSLayer` is located on the chip.
The `DynapcnnLayer` layers on the other hand have multiple potential core locations.
`chip_layers_ordering` helps specify where each of these layers is concretely placed.


Conversion between raster and spike streams
-------------------------------------------

You can use the convenience methods `raster_to_events()` or `xytp_to_events()` of the `ChipFactory` to generate `Spike` sequences of the appropriate type.


Samna: The interface library to the chip
----------------------------------------
SynSense develops Samna, a library that handles the communication and configuration of the chip.
You will find further examples and API reference of Samna on its documentation page.
Documentation available `here <https://synsense-sys-int.gitlab.io/samna/index.html>`_.

