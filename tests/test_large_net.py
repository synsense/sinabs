"""
This module is meant to test a real use case. It will include testing of
the network equivalence, and of the correct output configuration.
"""

import samna

samna
# this is necessary as a workaround because of a problem
# that occurs when samna is imported after torch

from torch import nn
import torch
from sinabs.layers import NeuromorphicReLU
from sinabs.from_torch import from_model
from sinabs.backend.speck.tospeck import SpeckCompatibleNetwork


class SpeckNetA(nn.Module):
    def __init__(self, quantize=False, n_out=1):
        super().__init__()

        self.seq = [
            # core 0
            nn.Conv2d(2, 16, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False),
            NeuromorphicReLU(quantize=quantize, fanout=144),
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


sdc = SpeckNetA()
snn = from_model(sdc)

input_shape = (2, 128, 128)
input_data = torch.rand((1, *input_shape)) * 1000
snn.eval()
snn_out = snn(input_data)  # forward pass

snn.reset_states()
speck_net = SpeckCompatibleNetwork(snn, input_shape=input_shape, discretize=False)
speck_out = speck_net(input_data)

speck_config = speck_net.make_config(speck_layers_ordering=[0, 1, 2, 7, 4, 5, 6, 3, 8])


print("Snn out", snn_out.sum().item())
print("Speck out", speck_out.sum().item())

# - Make sure that layers of different models are distinct objects
for lyr_snn, lyr_speck in zip(snn.spiking_model.seq, speck_net.sequence):
    assert lyr_snn is not lyr_speck
