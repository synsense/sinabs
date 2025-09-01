import pytest
import torch
from torch import nn

from sinabs import layers as sl
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.backend.dynapcnn.exceptions import (
    InvalidGraphStructure,
    UnsupportedLayerType,
)
from sinabs.from_torch import from_model

@pytest.mark.skip("Need NONSEQ update")
@pytest.mark.parametrize("device", tuple(ChipFactory.supported_devices.keys()))
def test_too_large(device):
    # Model that is too big to fit on any of our architectures
    big_ann = nn.Sequential(
        nn.Conv2d(1, 3, 5, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Conv2d(3, 1, 5, 1, bias=False),
        nn.ReLU(),
        nn.AvgPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(16, 999999, bias=False),
    )
    input_shape = (1, 28, 28)

    hardware_incompatible_model = DynapcnnNetwork(
        from_model(big_ann, add_spiking_output=True, batch_size=1).cpu(),
        discretize=True,
        input_shape=input_shape,
    )

    assert not hardware_incompatible_model.is_compatible_with(device)

    with pytest.raises(ValueError):
        hardware_incompatible_model.to(device)

@pytest.mark.skip("Need NONSEQ update")
def test_missing_spiking_layer():
    in_shape = (2, 28, 28)
    snn = nn.Sequential(
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.IAFSqueeze(batch_size=1),
        sl.SumPool2d(2),
        nn.AvgPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
        sl.IAFSqueeze(batch_size=1),
        nn.Dropout2d(),
        nn.Conv2d(16, 2, kernel_size=3, stride=1, bias=False),
        sl.IAFSqueeze(batch_size=1),
        nn.Flatten(),
        nn.Linear(8, 5),
    )

    with pytest.raises(InvalidGraphStructure):
        net = DynapcnnNetwork(snn, input_shape=in_shape)

@pytest.mark.skip("Need NONSEQ update")
def test_incorrect_model_start():
    in_shape = (2, 28, 28)
    snn = nn.Sequential(
        sl.IAFSqueeze(batch_size=1),
        sl.SumPool2d(2),
        nn.AvgPool2d(2),
    )

    with pytest.raises(InvalidGraphStructure):
        net = DynapcnnNetwork(snn, input_shape=in_shape)


unsupported_layers = [
    nn.ReLU(),
    nn.Sigmoid(),
    nn.Tanh(),
    sl.LIF(tau_mem=5),
    sl.LIFSqueeze(batch_size=1, tau_mem=5),
    sl.NeuromorphicReLU(),
    sl.Cropping2dLayer(),
]

@pytest.mark.skip("Need NONSEQ update")
@pytest.mark.parametrize("layer", unsupported_layers)
def test_unsupported_layers(layer):
    in_shape = (1, 28, 28)
    ann = nn.Sequential(
        nn.Conv2d(1, 3, 5, 1, bias=False),
        layer,
    )

    with pytest.raises(UnsupportedLayerType):
        net = DynapcnnNetwork(ann, input_shape=in_shape)
