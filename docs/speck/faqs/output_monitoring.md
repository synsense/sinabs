# Output Monitoring

## How to Read The Output from Hidden Layers.

Basically, we can read all 9 layers output. The `CnnLayerConfig` of the `Samna Configuration` object has a boolean attribute 
called `monitor_enable`. If you want to read a specific layer's output, you must set the `monitor_enable` as `True` for
that layer.

By setting multiple layers' `monitor_enable` as True, we will receive `samna.xxx.event.Spike` events from 
the output buffer. Each `samna.xxx.event.Spike` has a `layer` attribute to indicate which layer it comes from.

```python
from torch import nn
from sinabs.layers import IAFSqueeze
from sinabs.backend.dynapcnn import DynapcnnNetwork
import samna

SNN = nn.Sequential(

    nn.Conv2d(1, 2, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),

    nn.Conv2d(2, 2, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),

    nn.Conv2d(2, 4, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),
)

"""
1. you can manually set the hidden layers monitor_enable as True
""" 
dynapcnn = DynapcnnNetwork(snn=SNN, input_shape=(1, 16, 16), dvs_input=False)
samna_cfg = dynapcnn.make_config(device="speck2fmodule")

hidden_layers = [0, 1]
for h_layer in hidden_layers:
    samna_cfg.cnn_layers[h_layer].monitor_enable = True

# check if the all layer's output can be read
# by default, we will monitor last layer's output
for layer in [0, 1, 2]:
    print(samna_cfg.cnn_layers[layer].monitor_enable)

# finally we just need to apply the samna configuration to the devkit, we finish the deployment.
devkit = samna.device.open_device("Speck2fModuleDevKit")
devkit.get_model().apply_configuration(samna_cfg)


"""
2. We can use a more convenient API (recommended)
"""
dynapcnn = DynapcnnNetwork(snn=SNN, input_shape=(1, 16, 16), dvs_input=False)
dynapcnn.to(device="speck2fmodule", monitor_layers=[0, 1, 2])
```


## How to Read The Vmem of the Neurons

Samna provide a type of event called `samna.xxx.event.ReadNeuronValue` to help us read the membrane potential of neurons.
Here we give an example on Speck2f.

```python
from torch import nn
from sinabs.layers import IAFSqueeze
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.io import calculate_neuron_address, neuron_address_to_cxy
import samna

SNN = nn.Sequential(

    nn.Conv2d(1, 2, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),

    nn.Conv2d(2, 2, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),

    nn.Conv2d(2, 4, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),
)

# deploy the snn to the devkit
dynapcnn = DynapcnnNetwork(snn=SNN, input_shape=(1, 16, 16), dvs_input=False)
dynapcnn.to(device="speck2fmodule")

# create ReadNeuronValue events as input
input_events = []

layers = [0, 1, 2]
x_list = [1, 1, 2]
y_list = [1, 1, 2]
channel = 0
for layer, x, y in zip(layers, x_list, y_list):
    address = calculate_neuron_address(x=x, y=y, c=channel, feature_map_size=(2, 16, 16))
    ev = samna.speck2f.event.ReadNeuronValue()
    ev.address = address
    ev.layer = layer
    input_events.append(ev)

# send the input to the devkit
output_events = dynapcnn(input_events)

# check the output 
for ev in output_events:

    address = ev.address
    c, x, y = neuron_address_to_cxy(address, feature_map_size=(2, 16, 16))
    layer = ev.layer
    vmem = ev.neuron_state
    print(f"vmem of layer {layer} at channel={c}, x={x}, y={y} is {vmem}!")
```