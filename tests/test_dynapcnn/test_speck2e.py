import numpy as np
import torch.nn as nn

from sinabs import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.dvs_layer import DVSLayer
from sinabs.backend.dynapcnn.dynapcnn_layer import DynapcnnLayer


def test_no_kill_bits():
    ann = nn.Sequential(
        nn.Conv2d(2, 10, kernel_size=3, bias=True),
        nn.ReLU(),
        nn.Conv2d(10, 3, kernel_size=2, bias=False),
        nn.ReLU(),
    )
    ann[0].weight.data[:] = 0
    snn = from_model(ann, input_shape=(2, 10, 10), batch_size=1)
    network = DynapcnnNetwork(snn=snn, input_shape=(2, 10, 10), dvs_input=True)

    dynapcnndevkit_config = network.make_config(
        chip_layers_ordering=[0, 1], device="dynapcnndevkit:0"
    )
    # Check that for weight values equal to zero, the kill bit is enabled
    assert (
        np.sum(dynapcnndevkit_config.cnn_layers[0].weights_kill_bit) == 2 * 10 * 3 * 3
    )

    speck2e_config = network.make_config(device="speck2edevkit:0")
    # Config is generated successfully even when speck2e does not have a kill bit


def test_speck2e_coordinates():
    """Generate the configuration for speck2edevkit."""
    ann = nn.Sequential(nn.Conv2d(2, 6, 3), nn.ReLU())
    snn = from_model(ann, input_shape=(2, 10, 10), batch_size=1)

    network = DynapcnnNetwork(snn, input_shape=(2, 10, 10), dvs_input=True)
    config = network.make_config(device="speck2edevkit:0")
    print(config.to_json())


def test_dvs_layer_generation():
    """DVSLayer should be generated is dvs input is enabled even for an empty network."""
    ann = nn.Sequential()
    network = DynapcnnNetwork(nn.Sequential(), input_shape=(2, 10, 10), dvs_input=True)
    assert isinstance(network.sequence[0], DVSLayer)
