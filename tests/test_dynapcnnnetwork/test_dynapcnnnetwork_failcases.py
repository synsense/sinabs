# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import pytest
import torch
from sianbs.backend.dynapcnn.chip_factory import ChipFactory

from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork

from .conftest_dynapcnnnetwork import args_DynapcnnNetworkTest


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

    hardware_incompatible_model = DynapcnnNetwork(
        from_model(big_ann, add_spiking_output=True, batch_size=1).cpu(),
        discretize=True,
        input_shape=input_shape,
    )

    assert not hardware_incompatible_model.is_compatible_with(device)

    with pytest.raises(ValueError):
        hardware_incompatible_model.to(device)
