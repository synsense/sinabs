import torch.nn as nn
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork

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
    dynapcnn_net = DynapcnnNetwork(sinabs_net.spiking_model, input_shape=input_shape, dvs_input=True)
    return dynapcnn_net


def test_chip_level_monitoring_enable():
    builder = ChipFactory("speck2b:0").get_config_builder()
    # Get the default config
    config = builder.get_default_config()

    # Enable monitors for some random set of layers
    builder.monitor_layers(config, ["dvs", 0, 5, 8])

    assert config.dvs_layer.monitor_enable == True
    assert config.cnn_layers[0].monitor_enable == True
    assert config.cnn_layers[5].monitor_enable == True
    assert config.cnn_layers[8].monitor_enable == True


def test_default_monitoring():
    dynapcnn_net = build_model()
    builder = ChipFactory("speck2b:0").get_config_builder()

    # As a default the last layer should be monitored
    config = dynapcnn_net.make_config(device="speck2b:0")
    clo = dynapcnn_net.chip_layers_ordering
    assert len(clo) > 0
    # Check that monitoring is off for all layers except last
    for layer in clo[:-1]:
        if layer == "dvs":
            assert config.dvs_layer.monitor_enable == False
        else:
            assert config.cnn_layers[layer].monitor_enable == False
    assert config.cnn_layers[clo[-1]].monitor_enable == True


def test_model_level_monitoring_enable():
    dynapcnn_net = build_model()
    builder = ChipFactory("speck2b:0").get_config_builder()

    # No layers are to be monitored
    config = dynapcnn_net.make_config(device="speck2b:0", monitor_layers=[])
    all_layers = list(range(9))
    for layer in all_layers:
        assert config.cnn_layers[layer].monitor_enable == False

    # Specify layers to monitor
    config = dynapcnn_net.make_config(device="speck2b:0", monitor_layers=["dvs", 5, -1])
    clo = dynapcnn_net.chip_layers_ordering
    assert len(clo) > 0

    assert config.dvs_layer.monitor_enable == True
    assert config.cnn_layers[clo[5]].monitor_enable == True
    assert config.cnn_layers[clo[-1]].monitor_enable == True
