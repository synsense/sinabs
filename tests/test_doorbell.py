"""
This module is meant to test a real use case. It will include testing of
the network equivalence, and of the correct output configuration.
"""
try:
    import samna
    samna
    TEST_CONFIGS = True
except ImportError:
    TEST_CONFIGS = False
# this is necessary as a workaround because of a problem
# that occurs when samna is imported after torch

from torch import nn
import torch
from sinabs.layers import NeuromorphicReLU
from sinabs.from_torch import from_model
from sinabs.backend.speck.tospeck import SpeckCompatibleNetwork
import pytest


class SmartDoorClassifier(nn.Module):
    def __init__(
        self, quantize=False, linear_size=32, n_channels_in=2, n_channels_out=1,
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
snn = from_model(sdc)

input_shape = (2, 64, 64)
input_data = torch.rand((1, *input_shape)) * 1000
snn.eval()
snn_out = snn(input_data)  # forward pass

snn.reset_states()
speck_net = SpeckCompatibleNetwork(snn, input_shape=input_shape, discretize=False)
speck_out = speck_net(input_data)


def test_same_result():
    print(speck_out)
    assert torch.equal(speck_out.squeeze(), snn_out.squeeze())


def test_auto_config():
    # - Should give an error with the normal layer ordering
    speck_net.make_config(speck_layers_ordering=[0, 1, 2, 3, 4])


def test_was_copied():
    # - Make sure that layers of different models are distinct objects
    for lyr_snn, lyr_speck in zip(snn.spiking_model.seq, speck_net.sequence):
        assert lyr_snn is not lyr_speck


if not TEST_CONFIGS:
    test_auto_config = pytest.mark.skip(test_auto_config)
