"""This module is meant to test a real use case.

It will include testing of the network equivalence, and of the correct output configuration.
"""

import pytest
import samna
import torch
from nirtorch.utils import sanitize_name
from torch import nn

from sinabs.backend.dynapcnn.dynapcnn_network import DynapcnnNetwork
from sinabs.from_torch import from_model
from sinabs.layers import NeuromorphicReLU

# this is necessary as a workaround because of a problem
# that occurs when samna is imported after torch


class SmartDoorClassifier(nn.Module):
    def __init__(
        self,
        quantize=False,
        linear_size=32,
        n_channels_in=2,
        n_channels_out=1,
    ):
        super().__init__()

        self.seq = [
            nn.Conv2d(
                in_channels=n_channels_in,
                out_channels=8,
                kernel_size=(3, 3),
                bias=False,
            ),
            NeuromorphicReLU(quantize=quantize, fanout=108),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=108),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=linear_size),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout2d(0.5),
            nn.Flatten(),
            nn.Linear(432, linear_size, bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=1),
            nn.Linear(linear_size, n_channels_out, bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=0),
        ]

        self.seq = nn.Sequential(*self.seq)

    def forward(self, x):
        return self.seq(x)


sdc = SmartDoorClassifier()
snn = from_model(sdc.seq, batch_size=1)

input_shape = (2, 64, 64)
input_data = torch.rand((1, *input_shape)) * 1000
snn.eval()
snn_out = snn(input_data)  # forward pass

snn.reset_states()
dynapcnn_net = DynapcnnNetwork(snn, input_shape=input_shape, discretize=False)
dynapcnn_out = dynapcnn_net(input_data)


def test_same_result():
    # print(dynapcnn_out)
    assert torch.equal(dynapcnn_out.squeeze(), snn_out.squeeze())


def test_auto_config():
    dynapcnn_net = DynapcnnNetwork(snn, input_shape=input_shape, discretize=True)
    dynapcnn_net.make_config(chip_layers_ordering=[0, 1, 2, 3, 4])
    dynapcnn_net.make_config(layer2core_map="auto")


def test_was_copied():
    # - Make sure that layers of different models are distinct objects
    # "Sanitize" all layer names, for compatibility with older nirtorch versions
    snn_layers = {
        sanitize_name(name): lyr for name, lyr in snn.spiking_model.named_modules()
    }
    idx_2_name_map = {
        idx: sanitize_name(name) for name, idx in dynapcnn_net.name_2_indx_map.items()
    }
    for idx, lyr_info in dynapcnn_net._graph_extractor.dcnnl_info.items():
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
