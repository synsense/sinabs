Advanced
========

Under the hood
--------------

The DynapcnnCompatibleNetwork converts a given model into a sequence of DVSLayer (at most 1) and DynapcnnLayers.

.. mermaid::

    graph TD
        DynapcnnCompatibleNetwork --> DVSLayer
        DynapcnnCompatibleNetwork --> DynapcnnLayer



A `ConfigBuilder` then converts this model to a config object when `make_config` method is called. 
`make_config` is called internally when a user calls the `to` method.

.. mermaid::

    graph LR
        DynapcnnCompatibleNetwork --> ConfigBuilder --> Samna Config

`ChipFactory` is used to fetch the appropriate `ConfigBuilder` for a given `device`.

.. mermaid::

    ChipFactory --> ConfigBuilder


The config object
-----------------

Memory constraints
------------------

Attributes of interest
----------------------

Know that the mapping of the various layers of the model to the layers of the chip is crucial.
`DynapcnnCompatibleNetwork.chip_layers_ordering` is a list of chip layer indices where a model was mapped.
This is useful when generating or interpreting events from `samna`, where the `layer` attribute refers to the layer on the chip.


Conversion between raster and spike streams
-------------------------------------------

You can use the convenience methods `raster_to_events()` or `xytp_to_events()` of the `ChipFactory` to generate `Spike` sequences of the appropriate type.

Testing model performance (device-independent)
----------------------------------------------


