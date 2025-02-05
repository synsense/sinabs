"""
Implementing "a network with a merge and a split" in https://github.com/synsense/sinabs/issues/181
"""

import torch.nn as nn

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.layers import IAFSqueeze, SumPool2d

dcnnl_map_2 = {
    0: {
        "input_shape": (2, 34, 34),
        "rescale_factors": set(),
        "is_entry_node": True,
        "conv": {
            "module": nn.Conv2d(2, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 0,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=8,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 1,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 1,
                "output_shape": (4, 33, 33),
            },
        ],
    },
    1: {
        "input_shape": (4, 33, 33),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 2,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=8,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 3,
        },
        "destinations": [
            {
                "pooling_ids": [4],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False),
                ],
                "destination_layer": 2,
                "output_shape": (4, 16, 16),
            },
            {
                "pooling_ids": [4],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False),
                ],
                "destination_layer": 3,
                "output_shape": (4, 16, 16),
            },
        ],
    },
    2: {
        "input_shape": (4, 16, 16),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 5,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=8,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 7,
        },
        "destinations": [
            {
                "pooling_ids": [8],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False),
                ],
                "destination_layer": 4,
                "output_shape": (4, 7, 7),
            },
        ],
    },
    3: {
        "input_shape": (4, 16, 16),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 6,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=8,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 11,
        },
        "destinations": [
            {
                "pooling_ids": [12],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False),
                ],
                "destination_layer": 6,
                "output_shape": (4, 7, 7),
            },
        ],
    },
    4: {
        "input_shape": (4, 7, 7),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 9,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=8,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 10,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 5,
                "output_shape": (4, 6, 6),
            },
        ],
    },
    5: {
        "input_shape": (4, 6, 6),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Linear(in_features=144, out_features=10, bias=False),
            "node_id": 15,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=8,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 16,
        },
        "destinations": [],
    },
    6: {
        "input_shape": (4, 7, 7),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 13,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=8,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 14,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 5,
                "output_shape": (4, 6, 6),
            },
        ],
    },
}

expected_output_2 = {
    0: {
        "input_shape": (2, 34, 34),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    1: {
        "input_shape": (4, 33, 33),
        "pool": [[2, 2], [2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    2: {
        "input_shape": (4, 16, 16),
        "pool": [[2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    3: {
        "input_shape": (4, 16, 16),
        "pool": [[2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    4: {
        "input_shape": (4, 7, 7),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    5: {
        "input_shape": (4, 6, 6),
        "pool": [],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    6: {
        "input_shape": (4, 7, 7),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    "entry_points": {0},
    "destination_map": {
        0: [1],
        1: [2, 3],
        2: [4],
        3: [6],
        4: [5],
        6: [5],
        5: [],
    },
}
