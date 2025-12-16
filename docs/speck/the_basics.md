The Basics
==========

Deploying an SNN on DYNAPCNN-based devices involves several steps, such as model architecture conversion, placement, and parameter quantization.
Sinabs automates this process for the end user and enables quick deployment and testing of your models on the devkits.

TLDR;
-----
A short (and perhaps the quickest path) to deploying your model on one of our chips is shown in the example below.

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
hw_model.to(device="speck2fdevkit:0")

# Send events to chip
events_in: List["Spike"] =  ... # Load your events
events_out = hw_model(events_in)

# Analyze your output events
...

```

Model conversion to DYNAP-CNN core structure
--------------------------------------------

Speck family chips, based on DYNAPCNN, comprise several `cores` or `layers`.
Each of these `layers` comprises three functionalities:

    1. 2D Convolution
    2. Integrate and Fire Neurons
    3. Sum Pooling

Accordingly, the `DynapcnnLayer` class is a `sequential` model with three layers:

    1. conv_layer
    2. spk_layer
    3. pool_layer

To deploy a model on these chips, the network architecture must be converted into a sequence of *DynapcnnLayer*s.
The `DynapcnnNetwork` class automates this model conversion from a sequential `sinabs` spiking neural network into a sequence of *DynapcnnLayer*s. 
In addition, it discretizes/quantizes the parameters to 8 bits (per chip specifications).


Layer conversion
----------------

Often, the network architecture is comprised of layers such as `AvgPool2d`, `Flatten`, or `Linear`.
The chips do not support these layers in their original form and require some transformation.
For instance, while `AvgPool2d` works in simulations, spikes cannot really be averaged. Instead, `SumPool2d` is a better fit for spiking networks.
Similarly, a `Linear` layer can be replaced with `Conv2d` with a kernel size 1x1, such that it is compatible with `DynapcnnLayer`.
Instantiating `DynapcnnNetwork` handles all such conversions.

Parameter quantization
----------------------

The hardware supports fixed-point weights (8-bit for weights and 16-bit for membrane potentials, for instance).
The models trained in PyTorch typically use a floating-point representation of weights.
Setting `discretize=True` converts the model parameters from floating-point to fixed-point representation while preserving the highest possible precision.

Device selection
----------------

The device naming is inspired by `pytorch` device naming convention, i.e., `DEVICE_TYPE:INDEX`.
`speck2fdevkit:0` refers to the _first_ `Speck 2F DevKit` available.
If there are multiple devices of the same kind connected to the PC, then they are referred to by higher incremental indices.

To see all the recognized devices, please have a look at the `sinabs.backend.dynapcnn.io.device_types`

```python
from sinabs.backend.dynapcnn import io
print(io.device_types)
```

List of devices currently recognized by *samna*
-----------------------------------------------

.. note::
    Not all of these are supported by this plugin, and not all of these are compatible with DYNAP-CNN

```
# A map of all device types and their corresponding samna `device_name`
device_types = {
    "speck2e": "Speck2eTestBoard",
    "speck2edevkit": "Speck2eDevKit",
    "speck2fmodule": "Speck2fModuleDevKit",
    "speck2fdevkit": "Speck2fDevKit",
    "dynapse2": "DYNAP-SE2 DevBoard",
    "dynapse2_stack": "DYNAP-SE2 Stack",
    "davis346": "Davis 346",
    "davis240": "Davis 240",
    "dvxplorer": "DVXplorer",
}
```

You can also get a list of all supported devices currently connected/available by running the following lines:

```python
from sinabs.backend.dynapcnn import io
io.get_all_samna_devices()
```

Finally, to get a list of all supported devices that Sinabs supports and can port your models to, inspect `ChipFactory.supported_devices`.

```python
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
ChipFactory.supported_devices
```
 
Placement of layers on device cores
-----------------------------------

A sequence of *DynapcnnLayer*s (i.e., a model converted to `DynapcnnNetwork`) is ready to be mapped onto chip cores/layers.
This is done by placing each model layer onto a chip layer. The exact placement is specified by the parameter `chip_layers_ordering`.

This is an essential parameter because each model layer has a specific memory requirement for kernel parameters and neurons.
The chip layers are not homogeneous and have a limited amount of memory allocated to each.
Consequently, not all chip layers will be compatible with every layer in the model.

If `chip_layers_ordering` is set to `"auto"`, the network is mapped onto the chip using a placement algorithm.
If the algorithm cannot place the model on the chip, it will throw an error.

Some methods helpful for debugging if you run into problems are `ConfigBuilder.get_valid_mapping()` and the object `ConfigBuilder.get_constraints()`.

After successfully mapping a model, the `chip_layers_ordering` can be inspected by executing `DynapcnnNetwork.chip_layers_ordering`.


Porting model to device
-----------------------

`DynapcnnNetwork` class has an API similar to that of native `pytorch` and its `.to` method.
Similar to porting a model to CPU with `model.to("cpu")` and GPU with `model.to("cuda:0")`, you can also port your `DynapcnnCompatibleModel` to a chip with `model.to("speck2fdevkit:0")`.

You can also specify a few additional parameters, as shown below:

```python
hw_model.to(
    device="speck2fdevkit:0",
    chip_layers_ordering="auto",  # default value is "auto"
    monitor_layers=[-1],
    config_modifier=config_modifier,
)
```

As shown in the example above, you can specify which layers to monitor.
Note here that the layer indices are those of the model. For instance, -1 refers to the model's last layer.
In addition, advanced users can pass a `config_modifier`.
This is a `callable`, or a function that takes a config object and applies any custom configuration changes before writing it to the chip.

See the `__doc__` string for further details on each of these parameters.

Sending and receiving spikes
----------------------------

You can send a pre-defined sequence of events to the chip using the model's forward method, as you do with standard `pytorch` models.

```python
events_out = hw_model(events_in)
```

The `events_in` has to be a list of `Spike` objects corresponding to the chip in use. 
Similarly, `events_out` will also be a list of `Spike` objects corresponding to the events generated from the monitored layers.
Each `Spike` event has attributes `layer`, `x`, `y`, `feature`, and `timestamp`.


Monitoring layer activity
-------------------------

To monitor the spiking activity of a given layer, the corresponding layer has to be specified in the `monitor_layers` parameter.
In most use cases, you will want to monitor the activity of the last layer of the model, and so this parameter will be set to [-1].

Once enabled, all corresponding spikes will appear in the sequence of returned events from the chip.
The `samna_output_buffer` accumulates all events emitted by the chip, including those from the monitored layers.
The events from this buffer are read and returned to the user when the forward method is called.

Recording data from hardware
----------------------------

If you want to record data from the speck sensor for a given duration, a simple way to do so is by:

Instantiate a DynapcnnNetwork with a sequential model that only contains a DVSLayer.

```python
shape = (128, 128)
layers = [
    DVSLayer(
        input_shape=shape,
    ),
    ]
snn = nn.Sequential(*layers)

dynapcnn = DynapcnnNetwork(
    snn=snn, dvs_input=True, discretize=True
)
```

Deploy that network onto the chip with the `to` method, passing monitor_layers = ["dvs"]


```python
dynapcnn.to(device="speck2fdevkit:0", monitor_layers=["dvs"])

```

To record, call the forward method of the DynapcnnNetwork instance with a list containing a single dummy event whose timestamp equals the desired recording duration (in microseconds).


```python
factory = ChipFactory("speck2fdevkit")
Spike = factory.get_config_builder().get_samna_module().event.Spike

input_data = [Spike(timestamp=10),]
events_out = dynapcnn(input_data)

```

The return value will be a list of all events recorded from the DVS within the specified duration.


