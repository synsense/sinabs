import numpy as np
import torch.nn as nn
import pytest

from sinabs import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dvs_layer import DVSLayer


def test_speck2e_coordinates():
    """Generate the configuration for speck2edevkit."""
    ann = nn.Sequential(nn.Conv2d(2, 6, 3), nn.ReLU())
    snn = from_model(ann, input_shape=(2, 10, 10), batch_size=1)

    network = DynapcnnNetwork(snn, input_shape=(2, 10, 10), dvs_input=True)
    config = network.make_config(device="speck2edevkit:0")
    print(config.to_json())


def test_dvs_layer_generation():
    """DVSLayer should be generated is dvs input is enabled even for an empty network."""
    network = DynapcnnNetwork(nn.Sequential(), input_shape=(2, 10, 10), dvs_input=True)
    assert isinstance(network.dvs_layer, DVSLayer)
