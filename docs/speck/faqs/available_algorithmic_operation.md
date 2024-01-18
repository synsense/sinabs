# Available Operations

## Available `torch.nn.modules` for Devkits

Current we support:

- nn.Conv2d
  - convolution kernel size < **16**
  - convolution channel number < **1024**
  
- nn.AvgPool2d(will be converted to `SumPool2d` when deployed to devkit)
  - pooling kernel size = [1, 2, 4]
  
- nn.Linear(will be converted to equivalent convolution layer when deployed to devkit)
- nn.Flatten

## Available `sinabs.layers` for Devkits

- IAFSqueeze
- SumPool2d
  - pooling kernel size = [1, 2, 4]

## Neuron Leakage

Our devkit support a constant leakage on the membrane-potential of the IAF neuron. To achieve that, all you need to do is:

- Enable the `bias` term of the `nn.Conv2d` layer.
- Make sure the `bias` is lower than 0 after the training.
- Turn on the slow-clock of the devkit which driven the leakage functionality when deploy the SNN to the devkit.

For more details see [how to leak neuron](./notebooks/leak_neuron.ipynb)