"""
Implementing "a complex network structure" example in https://github.com/synsense/sinabs/issues/181
"""

import torch
import torch.nn as nn

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.layers import IAFSqueeze, Merge, SumPool2d


class SNN(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, 2, 1, bias=False)
        self.iaf1 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

        self.conv2 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf2 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool2 = SumPool2d(2, 2)

        self.conv3 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf3 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool3 = SumPool2d(2, 2)
        self.pool3a = SumPool2d(2, 2)

        self.conv4 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf4 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool4 = SumPool2d(2, 2)

        self.flat1 = nn.Flatten()
        self.flat2 = nn.Flatten()

        self.conv5 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf5 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool5 = SumPool2d(2, 2)

        self.fc2 = nn.Linear(49, 10, bias=False)
        self.iaf2_fc = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

        # -- merges --
        self.merge1 = Merge()
        self.merge2 = Merge()

    def forward(self, x):
        # conv 1 - A/0
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)

        # conv 2 - B/1
        conv2_out = self.conv2(iaf1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)

        # conv 3 - C/2
        conv3_out = self.conv3(iaf1_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)
        pool3a_out = self.pool3a(iaf3_out)

        # conv 4 - D/3
        merge1_out = self.merge1(pool2_out, pool3_out)
        conv4_out = self.conv4(merge1_out)
        iaf4_out = self.iaf4(conv4_out)
        pool4_out = self.pool4(iaf4_out)
        flat1_out = self.flat1(pool4_out)

        # conv 5 - E/4
        conv5_out = self.conv5(pool3a_out)
        iaf5_out = self.iaf5(conv5_out)
        pool5_out = self.pool5(iaf5_out)
        flat2_out = self.flat2(pool5_out)

        # fc 2 - F/5
        merge2_out = self.merge2(flat2_out, flat1_out)

        fc2_out = self.fc2(merge2_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)

        return iaf2_fc_out


channels = 2
height = 34
width = 34
batch_size = 2
input_shape = (channels, height, width)

snn = SNN(batch_size)

expected_output = {
    "dcnnl_edges": {
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
        ("input", 0),
    },
    "node_source_map": {
        0: {"input"},
        1: {0},
        2: {0},
        3: {1, 2},
        4: {2},
        5: {3, 4},
    },
    "destination_map": {
        0: {1, 2},
        1: {3},
        2: {3, 4},
        3: {5},
        4: {5},
        5: {-1},
    },
    "sorted_nodes": [0, 1, 2, 3, 4, 5],
    "output_shape": torch.Size([2, 10, 1, 1]),
    "entry_points": {0},
}

# Sometimes the layer that usually gets assgined ID1, gets ID2, and the
# layer with ID 3 gets ID 4. Therefore an alternative solution is defined.
# This is not a bug in sinabs itself but an issue with the test, becuase
# the IDs that the layers are assigned do not always have to be the same.
expected_output["alternative"] = {
    "dcnnl_edges": {
        (0, 1),
        (0, 2),
        (1, 3),
        (1, 4),
        (2, 4),
        (3, 5),
        (4, 5),
        ("input", 0),
    },
    "node_source_map": {
        0: {"input"},
        1: {0},
        2: {0},
        3: {1},
        4: {1, 2},
        5: {3, 4},
    },
    "destination_map": {
        0: {1, 2},
        1: {3, 4},
        2: {4},
        3: {5},
        4: {5},
        5: {-1},
    },
    "sorted_nodes": [0, 1, 2, 3, 4, 5],
    "output_shape": torch.Size([2, 10, 1, 1]),
    "entry_points": {0},
}
