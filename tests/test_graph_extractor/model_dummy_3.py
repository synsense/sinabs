"""
Implementing "two networks with merging outputs" in https://github.com/synsense/sinabs/issues/181
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
input_shape = (batch_size, channels, height, width)

torch.manual_seed(0)
input_dummy = torch.randn(input_shape)

expected_output = {
    "edges": {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (8, 18),
        (17, 18),
        (18, 19),
        (19, 20),
        (20, 21),
        (21, 22),
        (22, 23),
        (23, 24),
    },
    "name_2_indx_map": {
        "conv_A": 0,
        "iaf_A": 1,
        "conv_B": 2,
        "iaf_B": 3,
        "pool_B": 4,
        "conv_C": 5,
        "iaf_C": 6,
        "pool_C": 7,
        "flat_brach1": 8,
        "conv_D": 9,
        "iaf_D": 10,
        "conv_E": 11,
        "iaf_E": 12,
        "pool_E": 13,
        "conv_F": 14,
        "iaf_F": 15,
        "pool_F": 16,
        "flat_brach2": 17,
        "merge": 18,
        "fc1": 19,
        "iaf1_fc": 20,
        "fc2": 21,
        "iaf2_fc": 22,
        "fc3": 23,
        "iaf3_fc": 24,
    },
    "entry_nodes": {0, 9},
    "nodes_io_shapes": {
        0: {"input": torch.Size([2, 34, 34]), "output": torch.Size([4, 33, 33])},
        9: {"input": torch.Size([2, 34, 34]), "output": torch.Size([4, 33, 33])},
        1: {"input": torch.Size([4, 33, 33]), "output": torch.Size([4, 33, 33])},
        10: {"input": torch.Size([4, 33, 33]), "output": torch.Size([4, 33, 33])},
        2: {"input": torch.Size([4, 33, 33]), "output": torch.Size([4, 32, 32])},
        11: {"input": torch.Size([4, 33, 33]), "output": torch.Size([4, 32, 32])},
        3: {"input": torch.Size([4, 32, 32]), "output": torch.Size([4, 32, 32])},
        12: {"input": torch.Size([4, 32, 32]), "output": torch.Size([4, 32, 32])},
        4: {"input": torch.Size([4, 32, 32]), "output": torch.Size([4, 16, 16])},
        13: {"input": torch.Size([4, 32, 32]), "output": torch.Size([4, 16, 16])},
        5: {"input": torch.Size([4, 16, 16]), "output": torch.Size([4, 15, 15])},
        14: {"input": torch.Size([4, 16, 16]), "output": torch.Size([4, 15, 15])},
        6: {"input": torch.Size([4, 15, 15]), "output": torch.Size([4, 15, 15])},
        15: {"input": torch.Size([4, 15, 15]), "output": torch.Size([4, 15, 15])},
        7: {"input": torch.Size([4, 15, 15]), "output": torch.Size([4, 7, 7])},
        16: {"input": torch.Size([4, 15, 15]), "output": torch.Size([4, 7, 7])},
        8: {"input": torch.Size([4, 7, 7]), "output": torch.Size([196, 1, 1])},
        17: {"input": torch.Size([4, 7, 7]), "output": torch.Size([196, 1, 1])},
        18: {"input": torch.Size([196, 1, 1]), "output": torch.Size([196, 1, 1])},
        19: {"input": torch.Size([196, 1, 1]), "output": torch.Size([100, 1, 1])},
        20: {"input": torch.Size([100, 1, 1]), "output": torch.Size([100, 1, 1])},
        21: {"input": torch.Size([100, 1, 1]), "output": torch.Size([100, 1, 1])},
        22: {"input": torch.Size([100, 1, 1]), "output": torch.Size([100, 1, 1])},
        23: {"input": torch.Size([100, 1, 1]), "output": torch.Size([10, 1, 1])},
        24: {"input": torch.Size([10, 1, 1]), "output": torch.Size([10, 1, 1])},
    },
}

snn = SNN(batch_size)
