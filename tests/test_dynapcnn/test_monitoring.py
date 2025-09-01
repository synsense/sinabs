import warnings

import pytest
import torch.nn as nn

from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.from_torch import from_model


def build_model():
    ann = nn.Sequential(
        nn.Conv2d(2, 16, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
        nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Dropout2d(0.5),
        nn.Conv2d(64, 256, kernel_size=(2, 2), padding=(0, 0), bias=False),
        nn.ReLU(),
        nn.Dropout2d(0.5),
        nn.Conv2d(256, 128, kernel_size=(1, 1), padding=(0, 0), bias=False),
        nn.ReLU(),
        nn.Conv2d(128, 8, kernel_size=(1, 1), padding=(0, 0), bias=False),
        nn.ReLU(),
    )
    input_shape = (2, 128, 128)
    sinabs_net = from_model(ann, input_shape=input_shape, batch_size=1)
    dynapcnn_net = DynapcnnNetwork(
        sinabs_net.spiking_model, input_shape=input_shape, dvs_input=True
    )
    return dynapcnn_net


def test_chip_level_monitoring_enable():
    builder = ChipFactory("speck2edevkit:0").get_config_builder()
    # Get the default config
    config = builder.get_default_config()

    config.cnn_layers[8].destinations[0].pooling = 2

    # Enable monitors for some random set of layers - should warn becuase of pooling in layer 8
    with pytest.warns(Warning):
        builder.monitor_layers(config, ["dvs", 0, 5, 8])

    assert config.dvs_layer.monitor_enable == True
    assert config.cnn_layers[0].monitor_enable == True
    assert config.cnn_layers[5].monitor_enable == True
    assert config.cnn_layers[8].monitor_enable == True

    config.dvs_layer.pooling.x = 2
    # Should warn becuase of pooling in dvs layer
    with pytest.warns(Warning):
        builder.monitor_layers(config, ["dvs"])

@pytest.mark.skip("Need NONSEQ update")
def test_default_monitoring():
    dynapcnn_net = build_model()
    builder = ChipFactory("speck2edevkit:0").get_config_builder()

    # As a default the last layer should be monitored
    config = dynapcnn_net.make_config(device="speck2edevkit:0")
    l2c = dynapcnn_net.layer2core_map # TODO - old code: dynapcnn_net.chip_layers_ordering
    assert len(l2c) > 0
    # Check that monitoring is off for all layers except last
    for layer, core in l2c.items():
        if layer in dynapcnn_net.exit_layer_ids:
            assert config.cnn_layers[core].monitor_enable == True
        else:
            assert config.cnn_layers[core].monitor_enable == False

@pytest.mark.skip("Need NONSEQ update")
def test_model_level_monitoring_enable():
    dynapcnn_net = build_model()
    builder = ChipFactory("speck2edevkit:0").get_config_builder()

    # No layers are to be monitored
    config = dynapcnn_net.make_config(device="speck2edevkit:0", monitor_layers=[])
    all_layers = list(range(9))
    for layer in all_layers:
        assert config.cnn_layers[layer].monitor_enable == False

    # Specify layers to monitor - should warn becuase layer 5 has pooling
    with pytest.warns(Warning):
        config = dynapcnn_net.make_config(
            device="speck2edevkit:0", monitor_layers=["dvs", 5, -1]
        )
    l2c = dynapcnn_net.layer2core_map
    assert len(l2c) > 0

    assert config.dvs_layer.monitor_enable == True
    assert config.cnn_layers[l2c[5]].monitor_enable == True
    for idx in dynapcnn_net.exit_layer_ids:
        assert config.cnn_layers[l2c[idx]].monitor_enable == True

    # Specify layers to monitor - should not warn becuase final layer has no pooling
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        config = dynapcnn_net.make_config(device="speck2edevkit:0", monitor_layers=[-1])

    # Monitor all layers
    config = dynapcnn_net.make_config(device="speck2edevkit:0", monitor_layers="all")
    assert all(config.cnn_layers[i].monitor_enable == True for i in l2c.values())
