# Available Network Architecture

## Brief Introduction of the DynapCNN Core

Our devkit has 9 `DynapCNN Core`s, each core can executes `Asynchronous Convolution`.
Each core also have the following
features:

1. Each core has an unique index number, the first core's index starts from 0. So the index is in range [0, 8).

2. Each core can **only** define **2 destination cores** as its output destination core(layer).
 See [detail.](https://synsense-sys-int.gitlab.io/samna/reference/dynapcnn/configuration/index.html#samna.dynapcnn.configuration.CNNLayerDestination)

3. You can define multiple cores to have one **same destination core**.
Technically,you can use this feature to achieve a `short-cut`(residual connection) like ResNet does.
See [detail.](https://synsense-sys-int.gitlab.io/samna/reference/dynapcnn/configuration/index.html#samna.dynapcnn.configuration.CNNLayerDestination.layer)

4. Each core can optionally apply a `sum-pooling` operation before feeding the output events into the destination core.
See [detail.](https://synsense-sys-int.gitlab.io/samna/reference/dynapcnn/configuration/index.html#samna.dynapcnn.configuration.CNNLayerDestination.pooling)

5. Each core can optionally apply a `channel-shift` operation before feeding the output events into the destination core.
Technically, you can use this feature to `concatenate` two cores' output events among the channel axis.
See [detail.](https://synsense-sys-int.gitlab.io/samna/reference/dynapcnn/configuration/index.html#samna.dynapcnn.configuration.CNNLayerDestination.feature_shift)

6. You can define the order of cores/layers by defining the `destination` index of each core, i.e. the order of the 9 layers on the chip can be customized by yourself.
We already have this feature in sinabs-dynapcnn. When you deploy an SNN to the devkit, you can do:

```python
from sinabs.backend.dynapcnn import DynapcnnNetwork

# suppose snn is a 4-layers network
dynapcnn = DynapcnnNetwork(snn=snn, discretize=True, dvs_input=False, input_shape=input_shape)
# deploy the SNN
dynapcnn.to(devcie="your device", chip_layers_ordering="auto")
# or you can do
dynapcnn.to(devcie="your device", chip_layers_ordering=[0, 1, 2, 3])
# or
dynapcnn.to(devcie="your device", chip_layers_ordering=[2, 5, 7, 1])
```

## What network structure can I define?

Currently, `sinabs-dynapcnn` can only parse a `torch.nn.Sequential` like architecture. So it is recommended to
use a `Sequential` like network. We are developing a network graph extraction feature at the present, which will
help the user to deploy their networks with more complex architecture to the devkit.


## Can I achieve a "Residual Connection" like ResNet does?

Like mentioned above, "Yes, we can define a residual short-cut on the devkit". However, currently you can only manually
change the `samna.dynapcnn.configuration.CNNLayerDestination.layer` to achieve this, you can do this if you are very
familiar with the `samna-configuration`. Otherwise,let's wait for a while after the  "network graph extraction feature" is
completed.


## What execution order should I be aware of when I am implementing a sequential structure?
You should be aware with the internal layer order.
DYNAP-CNN techonology defines serveral layers that can be communicates each other.
In a layer, the Convolution and Neuron activation must be implemented with an order like:

**Conv--> IAF -->pool(optional)**

The cascaded convolution and neuron activation in a DYNAPCNN layer is not allowed.

![dataflow](../_static/Overview/dataflow_layers.png)

### Ex1. Bad Case: Cascaded convlution
```

network = nn.sequential([
                        nn.conv2d(),
                        nn.conv2d(),
                        IAFsqueeze(),
                        ])

```
### Ex2. Bad Case: None sequential
```

class Network:

    def __init__(self):
        self.conv1 = nn.conv2d()
        self.iaf = IAFsqueeze()
    def forward(self, x):
        out = self.conv1(x)
        out = self.iaf(out)
        return out

```

### Ex3. Bad Case: Use unsupport operation

```
network = nn.sequential([
                        nn.conv2d(),
                        nn.BatchNorm2d(), # unspport in speck/dynapcnn
                        IAFsqueeze(),
                        ])
```

### Ex3. Good Case: Use unsupport operation

```
network = nn.sequential([
                        nn.conv2d(),
                        IAFsqueeze(),
                        nn.pool(),
                        # up to here is using 1 dynapcnn layer
                        nn.conv2d(),
                        IAFsqueeze(),
                        nn.Flatten(),
                        nn.Linear(),
                        IAFsqueeze(),
                        # up to here is using 2 dynapcnn layer
                        ])
```

## Memory Constrains
Each core has a different "neuron memory" and "weight memory" constraints in the design.
Please be careful about the memory limitations when you design your SNN.
See detail in the [overview of devkit.](../getting_started/overview.md)

## Limitation of Using ReadoutLayer

If you are using readout layer, the number of output class should < **15**.
See detail in the [readout layer introduction.](../getting_started/notebooks/using_readout_layer.ipynb)

