The Basics
==========

Deploying a SNN on devices based on DYNAP-CNN technology involves several steps such as model architecture conversion, placement and parameter quantization.
This package automates this process for the end-user and enables quick deployment and testing of your models on to the dev-kits.

TLDR;
-----
A short (and perhaps the quickest path) to deploying your model on one of our dev-kits is shown in the example below.

```python
import torch
import torch.nn as nn
from typing import List
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork

ann = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)

# Load your weights or train the model
ann.load_state_dict(torch.load("model_params.pt"), map_location="cpu")

# Convert your model to SNN
sinabs_model = from_model(ann, add_spiking_output=True)  # Your sinabs SNN model

# Convert your SNN to `DynapcnnNetwork`
hw_model = DynapcnnNetwork(
    sinabs_model.spiking_model,
    discretize=True,
    input_shape=(1, 28, 28)
)

# Deploy model to a dev-kit
hw_model.to(device="dynapcnndevkit:0")

# Send events to chip
events_in: List["Spike"] =  ... # Load your events
events_out = hw_model(events_in)

# Analyze your output events
...

```

Model conversion to DYNAP-CNN core structure
--------------------------------------------

DYNAP-CNN based chips like `DYNAP-CNN DevKit` or `Speck` series comprise several `cores` or `layers`. 
Each of these `layers` comprises three functionalities:

    1. 2D Convolution
    2. Integrate and Fire Neurons
    3. Sum Pooling

Accordingly, the `DynapcnnLayer` class is a `sequential` model with three layers:

    1. conv_layer
    2. spk_layer
    3. pool_layer

In order to deploy a model onto these chips, the network structure needs to be converted into a sequence of *DynapcnnLayer*s.
The `DynapcnnNetwork` class automates this model conversion from a sequential `sinabs` spiking neural network into a sequence of *DynapcnnLayer*s. 
In addition, it also descretizes/quantizes the parameters to 8 bits (according to the chip specifications).


Layer conversion
----------------

Often, the network architectures comprise of layers such as `AvgPool2d`, `Flatten` or `Linear`.
The chips do not support these layers in their original form and require some transformation.
For instance, while `AvgPool2d` works in simulations, spikes cannot really be averaged. Instead `SumPool2d` is a better fit for spiking networks.
Similarly, a `Linear` layer can be replaced with `Conv2d` with a kernel size 1x1 such that it is compatible with `DynapcnnLayer`.
Instantiating `DynapcnnNetwork` takes care of all such conversions.

Parameter quantization
----------------------

The hardware suppports fixed point weights (8 bits for weights and 16 bits for membrane potentials for instance).
The models trained in pytorch typically use floating point representation of weights. 
Setting `discretize=True` converts the model parameters from floating point to fixed point representation while preserving the highest possible precision.

Device selection
----------------

The device naming is inspired by `pytorch` device naming convention ie `DEVICE_TYPE:INDEX`.
`dynapcnndevkit:0` refers to the _first_ `Dynapcnn DevKit` available. 
If there are multiple devices of the same kind connected to the PC, then they are referred by higher incremental indices.

To see all the recognized devices, please have a look at the `sinabs.backend.dynapcnn.io.device_types`

```python
from sinabs.backend.dynapcnn import io
print(io.device_types)
```

List of devices currently recognized by *samna*
-----------------------------------------------

.. note::
    Not all of these are supported by this plugin and not all of these are compatible with DYNAP-CNN

```
# A map of all device types and their corresponding samna `device_name`
device_types = {
    "speck": "speck",
    "dynapse2": "DYNAP-SE2 DevBoard",
    "dynapse2_stack": "DYNAP-SE2 Stack",
    "speck2devkit": "Speck2DevKit",
    "dynapse1devkit": "Dynapse1DevKit",
    "davis346": "Davis 346",
    "davis240": "Davis 240",
    "dvxplorer": "DVXplorer",
    "pollendevkit": "PollenDevKit",
    "dynapcnndevkit": "DynapcnnDevKit",
}
```

You can also get a list of all supported devices currently connected/available by running the following lines:

```python
from sinabs.backend.dynapcnn import io
io.get_all_samna_devices()
```

Finally to get a list of all the supported devices that this plugin supports and allows you to port your models, 
you can inspect `ChipFactory.supported_devices`

```python
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
ChipFactory.supported_devices
```
 
Placement of layers on device cores
-----------------------------------

A sequence of *DynapcnnLayer*s (i.e. a model that has been converted to `DynapcnnNetwork`) is ready to be mapped onto the chip cores/layers.
This is done by placing each layer of the model onto a layer on the chip. The exact placement is specified by the parameter `chip_layers_ordering`.

This is an important parameter because each layer of the model has a certain memory requirement for kernel parameters and neurons.
The chip layers are not homogenous and have a limited amount of memory allocated to each. 
Consequently, not all layers on the chip will be compatible with each layer in the model.

If the `chip_layers_ordering` is set to `"auto"`, the network is going to be mapped onto the chip based on a placement algorithm.
If the algorithm is unable to place the model onto the chip, it will throw an error message.

Some methods helpful for debugging if you run into problems are `ConfigBuilder.get_valid_mapping()` and the object `ConfigBuilder.get_constraints()`.

After successfully mapping a model, the `chip_layers_ordering` can be inspected by executing `DynapcnnNetwork.chip_layers_ordering`.


Porting model to device
-----------------------

`DynapcnnNetwork` class has an API similar to that of native `pytorch` and its `.to` method.
Similar to porting a model to cpu with `model.to("cpu")` and GPU with `model.to("cuda:0")` you can also port your `DynapcnnCompatibleModel` to a chip with `model.to("dynapcnndevkit:0")`.

You can also specify a few additional parameters as shown below.

```python
hw_model.to(
    device="dynapcnndevkit:0",
    chip_layers_ordering="auto",  # default value is "auto"
    monitor_layers=[-1],
    config_modifier=config_modifier,
)
```

As shown in the above example, you can specify which layers are to be monitored. 
Note here that the layer indices are that of the model. For instance -1 refers to the last layer of the model.
In addition, for advanced users, a `config_modifier` can be passed.
This is a `callable` or a function that takes a config object and does any custom setting changes before writing this on the chip.

See the `__doc__` string for further details on each of these parameters.

Sending and receiving spikes
----------------------------

You can send a pre-defined sequence of events to the chip using the model's forward method as you do with standard `pytorch` models.

```python
events_out = hw_model(events_in)
```

The `events_in` has to be a list of `Spike` objects corresponding to the chip in use. 
Similarly `events_out` will also be a list of `Spike` objects corresponding to the events generated from the monitored layers.
Each `Spike` event has attributes `layer`, `x`, `y`, `feature` and `timestamp`. 


Monitoring layer activity
-------------------------

In order to monitor the spiking activity of a given layer, the corresponding layer has to be specified in the `monitor_layers` parameter. 
In most use cases, you will want to monitor the activity of the last layer of the model and so this parameter will be set to [-1].

Once enabled, all the corresponding spikes will be found in the sequence of returned events from the chip.
The `samna_output_buffer` accumulates all the events sent out by the chip, including those from the monitored layers.
The events from this buffer are then read out and returned to the user on calling the forward method.
