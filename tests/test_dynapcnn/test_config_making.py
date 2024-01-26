from copy import deepcopy

import pytest
import torch
import torch.nn as nn

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.from_torch import from_model

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
    for devkit in [
        "dynapcnndevkit",
        "speck2btiny",
        "speck2e",
        "speck2edevkit",
        "speck2fmodule",
    ]:
        config = hardware_compatible_model.make_config("auto", device=devkit)
        for idx, lyr in enumerate(config.cnn_layers):
            initial_value = torch.tensor(lyr.neurons_initial_value)

            shape = initial_value.shape
            zeros = torch.zeros(shape, dtype=torch.int)

            assert (
                initial_value.all() == zeros.all()
            ), f"Initial values of layer{idx} neuron states is not zeros!"


small_ann = nn.Sequential(
    nn.Conv2d(1, 3, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Conv2d(3, 1, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(16, 2, bias=False),
)

small_hardware_compatible_model = DynapcnnNetwork(
    from_model(small_ann, add_spiking_output=True, batch_size=1).cpu(),
    discretize=True,
    input_shape=input_shape,
)


@pytest.mark.parametrize("device", tuple(ChipFactory.supported_devices.keys()))
def test_verify_working_config(device):
    assert small_hardware_compatible_model.is_compatible_with(device)


# Model that is too big to fit on any of our architectures
big_ann = deepcopy(ann)
big_ann.append(nn.ReLU())
big_ann.append(nn.Linear(10, 999999, bias=False))

hardware_incompatible_model = DynapcnnNetwork(
    from_model(big_ann, add_spiking_output=True, batch_size=1).cpu(),
    discretize=True,
    input_shape=input_shape,
)


@pytest.mark.parametrize("device", tuple(ChipFactory.supported_devices.keys()))
def test_verify_non_working_config(device):
    assert not hardware_incompatible_model.is_compatible_with(device)
