import torch.nn as nn
import torch
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork


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

sinabs_model = from_model(ann, add_spiking_output=True, batch_size=1)
input_shape = (1, 28, 28)

hardware_compatible_model = DynapcnnNetwork(
    sinabs_model.spiking_model.cpu(),
    discretize=True,
    input_shape=input_shape,
)


def test_zero_initial_states():
    config = hardware_compatible_model.make_config("auto", device="dynapcnndevkit")
    for idx, lyr in enumerate(config.cnn_layers):
        initial_value = torch.tensor(lyr.neurons_initial_value)

        shape = initial_value.shape
        zeros = torch.zeros(shape, dtype=torch.int)

        assert initial_value.all() == zeros.all(), f"Initial values of layer{idx} neuron states is not zeros!"