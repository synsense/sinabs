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
        self.pool3a = SumPool2d(5, 5)

        self.conv4 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf4 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )
        self.pool4 = SumPool2d(3, 3)

        self.flat1 = nn.Flatten()
        self.flat2 = nn.Flatten()

        self.conv5 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf5 = IAFSqueeze(
            batch_size=batch_size,
            min_v_mem=-1.0,
            spike_threshold=1.0,
            surrogate_grad_fn=PeriodicExponential(),
        )

        self.fc2 = nn.Linear(25, 10, bias=False)
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
        flat2_out = self.flat2(iaf5_out)

        # fc 2 - F/5
        merge2_out = self.merge2(flat2_out, flat1_out)

        fc2_out = self.fc2(merge2_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)

        return iaf2_fc_out


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
        (1, 3),
        (2, 4),
        (4, 5),
        (5, 6),
        (3, 7),
        (7, 8),
        (7, 9),
        (8, 6),
        (9, 10),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (16, 15),
        (10, 17),
        (17, 16),
        (18, 19),
        (6, 11),
        (15, 18),
    },
    "name_2_indx_map": {
        "conv1": 0,
        "iaf1": 1,
        "conv2": 2,
        "conv3": 3,
        "iaf2": 4,
        "pool2": 5,
        "merge1": 6,
        "iaf3": 7,
        "pool3": 8,
        "pool3a": 9,
        "conv5": 10,
        "conv4": 11,
        "iaf4": 12,
        "pool4": 13,
        "flat1": 14,
        "merge2": 15,
        "flat2": 16,
        "iaf5": 17,
        "fc2": 18,
        "iaf2_fc": 19,
    },
    "entry_nodes": {0},
    "nodes_io_shapes": {
        0: {"input": torch.Size([2, 34, 34]), "output": torch.Size([1, 33, 33])},
        1: {"input": torch.Size([1, 33, 33]), "output": torch.Size([1, 33, 33])},
        2: {"input": torch.Size([1, 33, 33]), "output": torch.Size([1, 32, 32])},
        3: {"input": torch.Size([1, 33, 33]), "output": torch.Size([1, 32, 32])},
        4: {"input": torch.Size([1, 32, 32]), "output": torch.Size([1, 32, 32])},
        7: {"input": torch.Size([1, 32, 32]), "output": torch.Size([1, 32, 32])},
        5: {"input": torch.Size([1, 32, 32]), "output": torch.Size([1, 16, 16])},
        8: {"input": torch.Size([1, 32, 32]), "output": torch.Size([1, 16, 16])},
        9: {"input": torch.Size([1, 32, 32]), "output": torch.Size([1, 6, 6])},
        6: {"input": torch.Size([1, 16, 16]), "output": torch.Size([1, 16, 16])},
        10: {"input": torch.Size([1, 6, 6]), "output": torch.Size([1, 5, 5])},
        11: {"input": torch.Size([1, 16, 16]), "output": torch.Size([1, 15, 15])},
        17: {"input": torch.Size([1, 5, 5]), "output": torch.Size([1, 5, 5])},
        12: {"input": torch.Size([1, 15, 15]), "output": torch.Size([1, 15, 15])},
        16: {"input": torch.Size([1, 5, 5]), "output": torch.Size([25, 1, 1])},
        13: {"input": torch.Size([1, 15, 15]), "output": torch.Size([1, 5, 5])},
        14: {"input": torch.Size([1, 5, 5]), "output": torch.Size([25, 1, 1])},
        15: {"input": torch.Size([25, 1, 1]), "output": torch.Size([25, 1, 1])},
        18: {"input": torch.Size([25, 1, 1]), "output": torch.Size([10, 1, 1])},
        19: {"input": torch.Size([10, 1, 1]), "output": torch.Size([10, 1, 1])},
    },
}

snn = SNN(batch_size)
