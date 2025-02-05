"""
implementing "a network with residual connections" example in https://github.com/synsense/sinabs/issues/181
"""

import torch
import torch.nn as nn

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.layers import IAFSqueeze, Merge


class SNN(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False)  # node 0
        self.iaf1 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )  # node 1
        self.pool1 = nn.AvgPool2d(3, 3)  # node 2
        self.pool1a = nn.AvgPool2d(4, 4)  # node 3

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)  # node 4
        self.iaf2 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )  # node 6

        self.conv3 = nn.Conv2d(10, 1, 2, 1, bias=False)  # node 8
        self.iaf3 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )  # node 9

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(49, 500, bias=False)  # node 10
        self.iaf4 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )  # node 11

        self.fc2 = nn.Linear(500, 10, bias=False)  # node 12
        self.iaf5 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )  # node 13

        self.adder = Merge()

    def forward(self, x):

        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)
        pool1_out = self.pool1(iaf1_out)
        pool1a_out = self.pool1a(iaf1_out)

        conv2_out = self.conv2(pool1_out)
        iaf2_out = self.iaf2(conv2_out)

        conv3_out = self.conv3(self.adder(pool1a_out, iaf2_out))
        iaf3_out = self.iaf3(conv3_out)

        flat_out = self.flat(iaf3_out)

        fc1_out = self.fc1(flat_out)
        iaf4_out = self.iaf4(fc1_out)
        fc2_out = self.fc2(iaf4_out)
        iaf5_out = self.iaf5(fc2_out)

        return iaf5_out


channels = 2
height = 34
width = 34
batch_size = 3
input_shape = (channels, height, width)

snn = SNN(batch_size)

expected_output = {
    "dcnnl_edges": {
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 3),
        (3, 4),
        ("input", 0),
    },
    "node_source_map": {
        0: {"input"},
        1: {0},
        2: {0, 1},
        3: {2},
        4: {3},
    },
    "destination_map": {
        0: {1, 2},
        1: {2},
        2: {3},
        3: {4},
        4: {-1},
    },
    "entry_points": {0},
    "sorted_nodes": [0, 1, 2, 3, 4],
    "output_shape": torch.Size([3, 10, 1, 1]),
}
