# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "two networks with merging outputs" in https://github.com/synsense/sinabs/issues/181

import torch.nn as nn

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.layers import IAFSqueeze, SumPool2d

dcnnl_map_3 = {
    0: {
        "input_shape": (2, 34, 34),
        "inferred_input_shapes": set(),
        "rescale_factors": set(),
        "is_entry_node": True,
        "conv": {
            "module": nn.Conv2d(2, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
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
                "output_shape": (4, 33, 33),
            },
        ],
    },
    1: {
        "input_shape": (4, 33, 33),
        "inferred_input_shapes": set(((4, 33, 33),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 2,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 2,
        },
        "destinations": [
            {
                "pooling_ids": [4],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False)
                ],
                "destination_layer": 2,
                "output_shape": (4, 16, 16),
            },
        ],
    },
    2: {
        "input_shape": (4, 16, 16),
        "inferred_input_shapes": set(((4, 16, 16),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 5,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 6,
        },
        "destinations": [
            {
                "pooling_ids": [7],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False)
                ],
                "destination_layer": 3,
                "output_shape": (4, 7, 7),
            },
        ],
    },
    3: {
        "input_shape": (196, 1, 1),
        "inferred_input_shapes": set(((4, 7, 7),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Linear(in_features=196, out_features=100, bias=False),
            "node_id": 17,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 18,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 7,
                "output_shape": (100, 1, 1),
            },
        ],
    },
    4: {
        "input_shape": (2, 34, 34),
        "inferred_input_shapes": set(),
        "rescale_factors": set(),
        "is_entry_node": True,
        "conv": {
            "module": nn.Conv2d(2, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 8,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 9,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 5,
                "output_shape": (4, 33, 33),
            },
        ],
    },
    5: {
        "input_shape": (4, 33, 33),
        "inferred_input_shapes": set(((4, 33, 33),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 10,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
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
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False)
                ],
                "destination_layer": 6,
                "output_shape": (4, 16, 16),
            },
        ],
    },
    6: {
        "input_shape": (4, 16, 16),
        "inferred_input_shapes": set(((4, 16, 16),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False),
            "node_id": 13,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 14,
        },
        "destinations": [
            {
                "pooling_ids": [15],
                "pooling_modules": [
                    SumPool2d(kernel_size=2, stride=2, ceil_mode=False)
                ],
                "destination_layer": 3,
                "output_shape": (4, 7, 7),
            },
        ],
    },
    7: {
        "input_shape": (100, 1, 1),
        "inferred_input_shapes": set(((100, 1, 1),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Linear(in_features=100, out_features=100, bias=False),
            "node_id": 19,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 20,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 8,
                "output_shape": (100, 1, 1),
            },
        ],
    },
    8: {
        "input_shape": (100, 1, 1),
        "inferred_input_shapes": set(((100, 1, 1),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Linear(in_features=100, out_features=10, bias=False),
            "node_id": 21,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=2,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 22,
        },
        "destinations": [],
    },
}

expected_output_3 = {
    0: {
        "input_shape": (2, 34, 34),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [1],
        "entry_node": True,
    },
    1: {
        "input_shape": (4, 33, 33),
        "pool": [[2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [2],
        "entry_node": False,
    },
    2: {
        "input_shape": (4, 16, 16),
        "pool": [[2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [3],
        "entry_node": False,
    },
    3: {
        "input_shape": (4, 7, 7),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [7],
        "entry_node": False,
    },
    4: {
        "input_shape": (2, 34, 34),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [5],
        "entry_node": True,
    },
    5: {
        "input_shape": (4, 33, 33),
        "pool": [[2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [6],
        "entry_node": False,
    },
    6: {
        "input_shape": (4, 16, 16),
        "pool": [[2, 2]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [3],
        "entry_node": False,
    },
    7: {
        "input_shape": (100, 1, 1),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [8],
        "entry_node": False,
    },
    8: {
        "input_shape": (100, 1, 1),
        "pool": [],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [],
        "entry_node": False,
    },
}
