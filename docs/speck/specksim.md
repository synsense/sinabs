# Specksim 

## Introduction

Specksim is a high performance Spiking-Convolutional Neural Network simulator which is written in C++ and bound to our backend library `samna`. It simulates the SNN completely event-based and is a good representation of how our `DYNAP-CNN` architecture hardware processes events. It is designed to help users who do not have access to our chips to test their networks to see how they would perform if they were deployed on the chip. Furthermore, the network architectures that normally could not have been deployed to our hardware due to memory constrains can be tested in `specksim`. Specksim package is completely bound to Python, which means that any and all of its features can be used easily from Python.

## Setup
`Specksim` is implemented partly in `sinabs` and partly in `samna`. If this library is installed via `pip` both of these libraries will be present. Therefore, no additional packages are necessary. 

## Supported architecture
The `specksim` simulator only supports sequential models. Typically the container module we use is `torch.nn.Sequential`, however nested modules are also supported.

Each weight layer, `torch.nn.Conv2d`, `torch.nn.Linear` should be followed by a spiking layer `sinabs.layers.IAF`, `sinabs.layers.IAFSqueeze`.

The output layer of the network has to be a `spiking layer`.

## Supported layers

### Parameter layers
This simulation supports two weight layer, namely `torch.nn.Conv2d` and `torch.nn.Linear`. Any other parameter layer will be discarded. Linear layer will be converted into a Conv2d layer keeping all its features. `Biases` are `not` supported. We currently do not have a way to simulate biases as they behave on our hardware.

### Pooling layers
The simulation supports two pooling layers, namely `torch.nn.AvgPool2d` and `sinabs.layers.SumPool2d`. Any other pooling layer will be discarded. AvgPool2d layer will be converted to a SumPool2d layer, and the weights of the following parameter layer will be scaled down based on the kernel size of the pooling layer. Padding is not supported. Stride is considered to be equal to the kernel size for simulating the on-chip behaviour. 

### Spiking layers
Only `sinabs.layers.IAF` and `sinabs.layers.IAFSqueeze` layers are supported. `LIF` neurons are not supported. 

## Input/Output
`Specksim` expects events in the format of `np.record` arrays. The record array has to have the following 4 keys. `x`, `y`, `t` and `p`; where `x` and `y` are the coordinates, `t` stands for timestamps and `p` stands for polarity (or in the context of non DVS input channels). The four keys are converted in `np.uint32` format. Output format which can be seen in the next cell can be used for the input format as well.

```
event_format = np.dtype([("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("p", np.uint32)])
```

## How-to-use

### Imports
```
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn.specksim import from_sequential
```

### Define an artificial neural network
```
ann = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)
```

You can then load weights.

### Convert to SNN using sinabs
```
snn = from_model(ann, add_spiking_output=True, batch_size=1).spiking_model
```

Now that a `sinabs` spiking neural network is defined, we can convert this model to a `SpecksimNetwork` using the method `from_sequential`. 

### Convert sequential SNNs to SpecksimNetwork

```
input_shape = (1, 28, 28) # MNIST shape
specksim_network_dynapcnn = from_sequential(snn, input_shape=input_shape)
```

Please note that the input shape of the network has to be passed explicitly.

### Convert from sequential DynapcnnNetwork to SpecksimNetwork

In order to do a more realistic simulation, the sequential SNN network can be quantized by converting it into a `DynapcnnNetwork` object, then the quantized `DynapcnnNetwork` can be converted into the `SpecksimNetwork`.

```
input_shape = (1, 28, 28) # MNIST shape
dynapcnn_network = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=False, discretize=True)
# the dynapcnn_network weights are quantized as we passed discretize=True
specksim_network_dynapcnn = from_sequential(dynapcnn_network, input_shape=input_shape)
```

### Send events to the simulated SNN

Now that the network is created we can send events and see if we receive any.

```
x = 10 # in pixels
y = 10 # in pixels
t = 100_000 # typically in microseconds
p = 0
input_event = np.array([x,y,t,p], dtype=specksim_network.output_dtype)
output_event = specksim_network(input_event)
print(output_event)
```

### Monitoring hidden layers

Hidden layers of the network can be monitored by adding monitors to certain spiking layers. Please note that only the spiking layers are allowed to be monitored. In order to use that we use the `SpecksimNetwork.add_monitor(int)` or in order to add monitors to multiple layers, we can use the following `SpecksimNetwork.add_monitors(List[int])`. For monitoring the `N`th spiking layer, you can pass `N`. Please note that the indices start from 0.  

```
specksim_network.add_monitor(0)
output_event = specksim_network(input_event)
```

Now that the forward pass has been completed, we can read from the monitor we have added. We do so using `SpecksimNetwork.read_monitor(int)`. In order to read from multiple monitors, you can use the following convenience methods. `SpecksimNetowrk.read_monitors(List[int])` or `SpecksimNetwork.read_all_monitors()` methods. Indexing is handled in the same way as adding monitors.

```
intermadiate_layer_events: np.record = specksim_network.read_monitor(0)
print(intermediate_layer_events)
```

Please note that after each forward pass the monitors are flushed.

### Monitoring hidden layer states.

Due to the event-based nature of the hardware and simulation, the state updates cannot be stored as there are too many changes to store in the memory. However, the states can be read at the end of each call with `read_spiking_layer_states(int)` method. If the argument is `N` then the states of `N`th spiking layer will be received.

```
states: List[List[List[int]]] = specksim_network.read_spiking_layer_states(0)
print(states)
```

### Resetting states
`Specksim` is mainly designed for benchmarking network performances in an event-driven way. In benchmarking we typically reset the spiking layer states. This can be achived by the following.

```
specksim_network.reset_states()
```

## Drawbacks and possible questions

### No training
This simulator is inference only and it does not support training.

### No biases
Biases for weight layers, namely `torch.nn.Conv2d` and `torch.nn.Linear` are not supported. The biases should be set to `False` explicitly in the network before conversion.

### Breadth-first vs Depth-first
Our `DYNAP-CNN` architecture is completely asynchronous. This means that for a sequential model, each event comes after the event that created it from the previous layer. That means the hardware processes the events in a `depth-first` manner. This simulation however processes events layer-by-layer. If there are multiple events received by a layer. That layer first finishes processing these events and then adds them to a queue, so the next layer can start their processing. This is done to make the simulation more efficient and faster. Furthermore, for models that were trained in a rate-based manner, the change in the processing scheme does not create too big of a difference.

TLDR; The chip processes events in `depth-first` manner, whereas due to implementation efficiency `specksim` processes events in `breadth-first` manner.

### Timestamping
Our hardware with the `DYNAP-CNN` architecture are completely asynchronous. Therefore, event timestamps do not play any role in calculation in the chip itself. The output events however do have timestamps, which are handled by the development boards. The timestamps depend on the load on the cores and therefore in a simulation cannot be replicated with a lot of success. Therefore, in this simulation for output events we assign the timestamp of the input event that led to its creation. 

### Real-time
It is not possible to reliably run spiking neural network in real-time using `specksim`. Although event-based processing is implemented completely in C++ for any complex architecture with a lot of parameters and fan-out there will be certain delays. 

## Try it yourself
An example of running a converted `SNN` trained on `MNIST` in `specksim` can be found under `examples/mnist/specksim_network.py`