"""
implementing "two networks with merging outputs" in https://github.com/synsense/sinabs/issues/181
"""

import torch
import torch.nn as nn

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.layers import IAFSqueeze, Merge, SumPool2d


class SNN(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()

        self.conv_A = nn.Conv2d(2, 4, 2, 1, bias=False)
        self.iaf_A = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

        self.conv_B = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_B = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool_B = SumPool2d(2, 2)

        self.conv_C = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_C = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool_C = SumPool2d(2, 2)

        self.conv_D = nn.Conv2d(2, 4, 2, 1, bias=False)
        self.iaf_D = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

        self.conv_E = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_E = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool_E = SumPool2d(2, 2)

        self.conv_F = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_F = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool_F = SumPool2d(2, 2)

        self.flat_brach1 = nn.Flatten()
        self.flat_brach2 = nn.Flatten()
        self.merge = Merge()

        self.fc1 = nn.Linear(196, 100, bias=False)
        self.iaf1_fc = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

        self.fc2 = nn.Linear(100, 100, bias=False)
        self.iaf2_fc = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

        self.fc3 = nn.Linear(100, 10, bias=False)
        self.iaf3_fc = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

    def forward(self, x):
        # conv 1 - A
        conv_A_out = self.conv_A(x)
        iaf_A_out = self.iaf_A(conv_A_out)
        # conv 2 - B
        conv_B_out = self.conv_B(iaf_A_out)
        iaf_B_out = self.iaf_B(conv_B_out)
        pool_B_out = self.pool_B(iaf_B_out)
        # conv 3 - C
        conv_C_out = self.conv_C(pool_B_out)
        iaf_C_out = self.iaf_C(conv_C_out)
        pool_C_out = self.pool_C(iaf_C_out)

        # ---

        # conv 4 - D
        conv_D_out = self.conv_D(x)
        iaf_D_out = self.iaf_D(conv_D_out)
        # conv 5 - E
        conv_E_out = self.conv_E(iaf_D_out)
        iaf_E_out = self.iaf_E(conv_E_out)
        pool_E_out = self.pool_E(iaf_E_out)
        # conv 6 - F
        conv_F_out = self.conv_F(pool_E_out)
        iaf_F_out = self.iaf_F(conv_F_out)
        pool_F_out = self.pool_F(iaf_F_out)

        # ---

        flat_brach1_out = self.flat_brach1(pool_C_out)
        flat_brach2_out = self.flat_brach2(pool_F_out)
        merge_out = self.merge(flat_brach1_out, flat_brach2_out)

        # FC 7 - G
        fc1_out = self.fc1(merge_out)
        iaf1_fc_out = self.iaf1_fc(fc1_out)
        # FC 8 - H
        fc2_out = self.fc2(iaf1_fc_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)
        # FC 9 - I
        fc3_out = self.fc3(iaf2_fc_out)
        iaf3_fc_out = self.iaf3_fc(fc3_out)

        return iaf3_fc_out


channels = 2
height = 34
width = 34
batch_size = 2
input_shape = (channels, height, width)

snn = SNN(batch_size)

expected_output = {
    "dcnnl_edges": {
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 7),
        (1, 3),
        (3, 5),
        (5, 6),
        (7, 8),
        ("input", 0),
        ("input", 1),
    },
    "node_source_map": {
        0: {"input"},
        2: {0},
        4: {2},
        6: {4, 5},
        1: {"input"},
        3: {1},
        5: {3},
        7: {6},
        8: {7},
    },
    "destination_map": {
        0: {2},
        2: {4},
        4: {6},
        6: {7},
        1: {3},
        3: {5},
        5: {6},
        7: {8},
        8: {-1},
    },
    "sorted_nodes": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    "output_shape": torch.Size([2, 10, 1, 1]),
    "entry_points": {0, 1},
}
