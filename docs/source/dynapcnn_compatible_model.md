The basics
==========

Deploying a SNN on devices based on DYNAP-CNN technology involves several steps such as model architecture conversion, placement and parameter quantization.
This package automates this process for the end-user and enables quick deployment and testing of your models on to the dev-kits.

TLDR;
-----
The quickest path to deploying your model on one of our dev-kits is show below.

```python
import torch
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnCompatibleNetwork

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

# Convert your SNN to `DynapcnnCompatibleNetwork`
hw_model = DynapcnnCompatibleNetwork(
    sinabs_model.spiking_model,
    discretize=True,
    input_shape=(1, 28, 28)
)

# Deploy model to a dev-kit
hw_model.to(device="dynapcnndevkit:0")



```

Model conversion to DYNAP-CNN core structure
--------------------------------------------

Layer conversion
----------------

Parameter quantization
----------------------

Device selection
----------------

Mapping layers to device cores
------------------------------

Porting model to device
-----------------------

Sending and receiving spikes
----------------------------

Monitoring layer activity
-------------------------

Advanced
========

Under the hood
--------------

The config object
-----------------

Attributes of interest
----------------------

Dangers
=======

Do not use biases. They deploy a lot of overhead on the chip in terms of computation and consequently power.
At the moment, we do not support deploying models with biases BUT *there are no warnings to tell you when you do use biases*. 
So watchout!
