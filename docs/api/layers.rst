layers
========
.. currentmodule:: sinabs.layers

Parent layers
-------------

.. autosummary::
    :toctree: generated/
    :template: class_layer.rst

    StatefulLayer
    SqueezeMixin

Spiking
-------
All spiking layers are subclasses of `StatefulLayer` and Squeeze layers additionally inherit from `SqueezeMixin`.

.. autosummary::
    :toctree: generated/
    :template: class_layer.rst

    IAF
    IAFSqueeze
    IAFRecurrent
    LIF
    LIFSqueeze
    LIFRecurrent
    ALIF
    ALIFRecurrent

Non-spiking
-----------
These layers are special cases of LIF layers.

.. autosummary::
    :toctree: generated/
    :template: class_layer.rst

    ExpLeak
    ExpLeakSqueeze

Pooling
-------
.. autosummary::
    :toctree: generated/
    :template: class_layer.rst

    SpikingMaxPooling2dLayer
    SumPool2d
 
Conversion from images / analog signals
---------------------------------------
.. autosummary::
    :toctree: generated/
    :template: class_layer.rst

    Img2SpikeLayer
    Sig2SpikeLayer

Auxiliary
---------
.. autosummary::
    :toctree: generated/
    :template: class_layer.rst

    Cropping2dLayer
    Repeat
    FlattenTime
    UnflattenTime

ANN layers
----------
.. autosummary::
    :toctree: generated/
    :template: class_layer.rst

    NeuromorphicReLU
    QuantizeLayer

