# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "a complex network structure" example in https://github.com/synsense/sinabs/issues/181 . """

import torch.nn as nn

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.layers import IAFSqueeze, SumPool2d

dcnnl_map_4 = {
    0: {
        "input_shape": (2, 34, 34),
        "rescale_factors": set(),
        "is_entry_node": True,
        "conv": {
            "module": nn.Conv2d(2, 1, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 0,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
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
                "output_shape": (1, 33, 33),
            },
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 2,
                "output_shape": (1, 33, 33),
            },
        ],
    },
    1: {
        "input_shape": (1, 33, 33),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 2,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 4,
        },
        "destinations": [
            {
                "pooling_ids": [5],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False)
                ],
                "destination_layer": 3,
                "output_shape": (1, 16, 16),
            },
        ],
    },
    2: {
        "input_shape": (1, 33, 33),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 3,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
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
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False)
                ],
                "destination_layer": 3,
                "output_shape": (1, 16, 16),
            },
            {
                "pooling_ids": [9],
                "pooling_modules": [
                    SumPool2d(kernel_size=5, stride=5, ceil_mode=False)
                ],
                "destination_layer": 4,
                "output_shape": (1, 6, 6),
            },
        ],
    },
    3: {
        "input_shape": (1, 16, 16),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 11,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 12,
        },
        "destinations": [
            {
                "pooling_ids": [13],
                "pooling_modules": [
                    SumPool2d(kernel_size=3, stride=3, ceil_mode=False)
                ],
                "destination_layer": 5,
                "output_shape": (1, 5, 5),
            },
        ],
    },
    4: {
        "input_shape": (1, 6, 6),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 10,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 12,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 5,
                "output_shape": (1, 5, 5),
            },
        ],
    },
    5: {
        "input_shape": (1, 5, 5),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Linear(in_features=25, out_features=10, bias=False),
            "node_id": 16,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 17,
        },
        "destinations": [],
    },
}

expected_output_4 = {
    0: {
        "input_shape": (2, 34, 34),
        "pool": [[1, 1], [1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    1: {
        "input_shape": (1, 33, 33),
        "pool": [[2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    2: {
        "input_shape": (1, 33, 33),
        "pool": [[2, 2], [5, 5]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    3: {
        "input_shape": (1, 16, 16),
        "pool": [[3, 3]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    4: {
        "input_shape": (1, 6, 6),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    5: {
        "input_shape": (1, 5, 5),
        "pool": [],
        "rescale_factor": 1,
        "rescale_factors": set(),
    },
    "entry_points": {0},
    "destination_map": {
        0: [1, 2],
        1: [3],
        2: [3, 4],
        3: [5],
        4: [5],
        5: [],
    },
}
