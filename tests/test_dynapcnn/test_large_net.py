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
from hw_utils import find_open_devices


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


def test_too_large():
    with pytest.raises(ValueError):
        # - Should give an error with the normal layer ordering
        dynapcnn_net.make_config(chip_layers_ordering=range(9))


def test_auto_config():
    # - Should give an error with the normal layer ordering
    dynapcnn_net.make_config(chip_layers_ordering="auto")


def test_was_copied():
    # - Make sure that layers of different models are distinct objects
    for lyr_snn, lyr_dynapcnn in zip(snn.spiking_model, dynapcnn_net.sequence):
        assert lyr_snn is not lyr_dynapcnn


def test_make_config():
    dynapcnn_net = DynapcnnNetwork(
        snn, input_shape=input_shape, discretize=False, dvs_input=False
    )
    dynapcnn_out = dynapcnn_net(input_data)

    config = dynapcnn_net.make_config(
        device="speck2edevkit:0", chip_layers_ordering=[0, 1, 2, 7, 4, 5, 6, 3, 8]
    )
    config = dynapcnn_net.make_config(
        device="speck2edevkit:0", chip_layers_ordering="auto"
    )


def test_to_device():
    dynapcnn_net = DynapcnnNetwork(
        snn, input_shape=input_shape, discretize=False, dvs_input=False
    )
    dynapcnn_out = dynapcnn_net(input_data)

    devices = find_open_devices()

    if len(devices) == 0:
        pytest.skip("A connected Speck device is required to run this test")

    for device_name, _ in devices.items():

        dynapcnn_net.to(
            device=device_name, chip_layers_ordering=[0, 1, 2, 7, 4, 5, 6, 3, 8]
        )

        # TODO: this test fails when using speck2e but not speck 2f.
        # This has been reported in Samna: https://www.wrike.com/workspace.htm?acc=6529583#/inbox/work_item/1674059530
        # Close device for safe exit
        from sinabs.backend.dynapcnn import io

        io.close_device(device_name)
        dynapcnn_net.to(device=device_name)


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

    converted_channels = extended_net.sequence[-1].conv_layer.out_channels

    assert (out_channels - 1) * 4 + 1 == converted_channels
