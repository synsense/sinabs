"""This module is meant to test a real use case.

It will include testing of the network equivalence, and of the correct output configuration.
"""

# import samna
# this is necessary as a workaround because of a problem
# that occurs when samna is imported after torch

import pytest
import torch
from torch import nn

from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork
from sinabs.from_torch import from_model
from sinabs.layers import NeuromorphicReLU


class DynapCnnNetA(nn.Module):
    def __init__(self, quantize=False, n_out=1):
        super().__init__()

        self.seq = [
            # core 0
            nn.Conv2d(
                2, 16, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False
            ),
            NeuromorphicReLU(quantize=quantize, fanout=144),
            nn.Identity(),
            # core 1
            nn.Conv2d(16, 16, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=288),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 2
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=288),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 7
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=576),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 4
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=576),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 5
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=1024),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            # core 6
            nn.Dropout2d(0.5),
            nn.Conv2d(64, 256, kernel_size=(2, 2), padding=(0, 0), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=128),
            # core 3
            nn.Dropout2d(0.5),
            nn.Conv2d(256, 128, kernel_size=(1, 1), padding=(0, 0), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=11),
            # core 8
            nn.Conv2d(128, n_out, kernel_size=(1, 1), padding=(0, 0), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=0),
            nn.Flatten(),  # otherwise torch complains
        ]

        self.seq = nn.Sequential(*self.seq)

    def forward(self, x):
        return self.seq(x)


sdc = DynapCnnNetA()
snn = from_model(sdc.seq, batch_size=1)

input_shape = (2, 128, 128)
input_data = torch.rand((1, *input_shape)) * 1000
snn.eval()
snn_out = snn(input_data)  # forward pass

snn.reset_states()
# NOTE: Test top_level_collect fails on dvs_input=False, but works if dvs_input=True
dynapcnn_net = DynapcnnNetwork(
    snn, input_shape=input_shape, discretize=False, dvs_input=False
)
dynapcnn_out = dynapcnn_net(input_data)


def test_same_result():
    # print(dynapcnn_out)
    assert torch.equal(dynapcnn_out.squeeze(), snn_out.squeeze())


# TODO: Define new test with actual network that is too large. Probably have it as fail case in test_dynapcnnnetwork
def test_too_large():
    pass


def test_auto_config():
    # - Should give an error with the normal layer ordering
    dynapcnn_net.make_config(chip_layers_ordering="auto")


def test_was_copied():
    from nirtorch.utils import sanitize_name

    # - Make sure that layers of different models are distinct objects
    snn_layers = {
        sanitize_name(name): lyr for name, lyr in snn.spiking_model.named_modules()
    }
    idx_2_name_map = {
        idx: sanitize_name(name) for name, idx in dynapcnn_net.name_2_indx_map.items()
    }
    for idx, lyr_info in dynapcnn_net._graph_extractor.dcnnl_map.items():
        conv_lyr_dynapcnn = dynapcnn_net.dynapcnn_layers[idx].conv_layer
        conv_node_idx = lyr_info["conv"]["node_id"]
        conv_name = idx_2_name_map[conv_node_idx]
        conv_lyr_snn = snn_layers[conv_name]
        assert conv_lyr_dynapcnn is not conv_lyr_snn

        spk_lyr_dynapcnn = dynapcnn_net.dynapcnn_layers[idx].spk_layer
        spk_node_idx = lyr_info["neuron"]["node_id"]
        spk_name = idx_2_name_map[spk_node_idx]
        spk_lyr_snn = snn_layers[spk_name]
        assert spk_lyr_dynapcnn is not spk_lyr_snn


def test_make_config():
    dynapcnn_net = DynapcnnNetwork(
        snn, input_shape=input_shape, discretize=False, dvs_input=False
    )
    dynapcnn_out = dynapcnn_net(input_data)

    config = dynapcnn_net.make_config(
        device="dynapcnndevkit:0", chip_layers_ordering=[0, 1, 2, 7, 4, 5, 6, 3, 8]
    )
    config = dynapcnn_net.make_config(
        device="dynapcnndevkit:0", chip_layers_ordering="auto"
    )


@pytest.mark.skip("Not suitable for automated testing. Depends on available devices")
def test_to_device():
    dynapcnn_net = DynapcnnNetwork(
        snn, input_shape=input_shape, discretize=False, dvs_input=False
    )
    dynapcnn_out = dynapcnn_net(input_data)

    dynapcnn_net.to(
        device="speck2b:0", chip_layers_ordering=[0, 1, 2, 7, 4, 5, 6, 3, 8]
    )

    # Close device for safe exit
    from sinabs.backend.dynapcnn import io

    io.close_device("speck2b:0")

    dynapcnn_net.to(device="speck2b:0")


def test_memory_summary():
    dynapcnn_net = DynapcnnNetwork(
        snn, input_shape=input_shape, discretize=False, dvs_input=False
    )
    summary = dynapcnn_net.memory_summary()

    print(summary)


@pytest.mark.parametrize("out_channels", [1, 2, 12])
def test_extended_readout_layer(out_channels: int):
    from sinabs.backend.dynapcnn.utils import extend_readout_layer

    sdc = DynapCnnNetA(n_out=out_channels)
    snn = from_model(sdc.seq, batch_size=1)

    input_shape = (2, 128, 128)

    # NOTE: Test top_level_collect fails on dvs_input=False, but works if dvs_input=True
    dynapcnn_net = DynapcnnNetwork(
        snn, input_shape=input_shape, discretize=False, dvs_input=False
    )
    extended_net = extend_readout_layer(dynapcnn_net)

    assert len(exit_layers := extended_net.exit_layers) == 1
    converted_channels = exit_layers[0].conv_layer.out_channels

    assert (out_channels - 1) * 4 + 1 == converted_channels
