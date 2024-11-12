# implementing sequential models

import torch
import torch.nn as nn

from sinabs.layers import IAFSqueeze, SumPool2d

input_shape_seq = (2, 30, 30)

seq_1 = nn.Sequential(
    nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
    IAFSqueeze(batch_size=1),
    nn.Conv2d(8, 2, kernel_size=3, stride=1, bias=False),
    IAFSqueeze(batch_size=1),
)

seq_2 = nn.Sequential(
    nn.Conv2d(2, 2, kernel_size=3, stride=1, bias=False),
    IAFSqueeze(batch_size=1),
    SumPool2d(2),
    nn.AvgPool2d(2),
    nn.Dropout(0.5),
    nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
    IAFSqueeze(batch_size=1),
    nn.Conv2d(8, 2, kernel_size=3, stride=1, bias=False),
    IAFSqueeze(batch_size=1),
    nn.Flatten(),
    nn.Linear(3 * 3 * 2, 5),
    nn.Identity(),
    IAFSqueeze(batch_size=1),
)

expected_seq_1 = {
    "dcnnl_edges": {
        ("input", 0),
        (0, 1),
    },
    "node_source_map": {
        0: {"input"},
        1: {0},
    },
    "destination_map": {
        0: {1},
        1: {-1},
    },
    "sorted_nodes": [0, 1],
    "output_shape": torch.Size([1, 2, 26, 26]),
    "entry_points": {0},
}

expected_seq_2 = {
    "dcnnl_edges": {
        (0, 1),
        (1, 2),
        (2, 3),
        ("input", 0),
    },
    "node_source_map": {
        0: {"input"},
        1: {0},
        2: {1},
        3: {2},
    },
    "destination_map": {
        0: {1},
        1: {2},
        2: {3},
        3: {-1},
    },
    "sorted_nodes": [0, 1, 2, 3],
    "output_shape": torch.Size([1, 5, 1, 1]),
    "entry_points": {0},
}
