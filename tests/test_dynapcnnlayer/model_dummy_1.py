# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "a network with residual connections" example in https://github.com/synsense/sinabs/issues/181

import torch.nn as nn

from sinabs.activation.surrogate_gradient_fn import PeriodicExponential
from sinabs.layers import IAFSqueeze

dcnnl_map_1 = {
    0: {
        "input_shape": (2, 34, 34),
        "inferred_input_shapes": set(),
        "rescale_factors": set(),
        "is_entry_node": True,
        "conv": {
            "module": nn.Conv2d(2, 10, kernel_size=(2, 2), stride=[1, 1], bias=False),
            "node_id": 0,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=3,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 1,
        },
        "destinations": [
            {
                "pooling_ids": [2],
                "pooling_modules": [nn.AvgPool2d(kernel_size=3, stride=3, padding=0)],
                "destination_layer": 1,
                "output_shape": (10, 11, 11),
            },
            {
                "pooling_ids": [3],
                "pooling_modules": [
                    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
                ],
                "destination_layer": 2,
                "output_shape": (10, 8, 8),
            },
        ],
    },
    1: {
        "input_shape": (10, 11, 11),
        "inferred_input_shapes": set(((10, 11, 11),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(10, 10, kernel_size=(4, 4), stride=[1, 1], bias=False),
            "node_id": 4,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=3,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 6,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 3,
                "output_shape": (10, 7, 7),
            },
        ],
    },
    2: {
        "input_shape": (10, 8, 8),
        "inferred_input_shapes": set(((10, 8, 8),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Conv2d(10, 1, kernel_size=(2, 2), stride=[1, 1], bias=False),
            "node_id": 7,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=3,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 8,
        },
        "destinations": [
            {
                "pooling_ids": [],
                "pooling_modules": [],
                "destination_layer": 3,
                "output_shape": (1, 7, 7),
            },
        ],
    },
    3: {
        "input_shape": (49, 1, 1),
        "inferred_input_shapes": set(((1, 7, 7),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Linear(in_features=49, out_features=500, bias=False),
            "node_id": 9,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=3,
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
                "destination_layer": 4,
                "output_shape": (500, 1, 1),
            },
        ],
    },
    4: {
        "input_shape": (500, 1, 1),
        "inferred_input_shapes": set(((500, 1, 1),)),
        "rescale_factors": set(),
        "is_entry_node": False,
        "conv": {
            "module": nn.Linear(in_features=500, out_features=10, bias=False),
            "node_id": 11,
        },
        "neuron": {
            "module": IAFSqueeze(
                batch_size=3,
                min_v_mem=-1.0,
                spike_threshold=1.0,
                surrogate_grad_fn=PeriodicExponential(),
            ),
            "node_id": 12,
        },
        "destinations": [],
    },
}

expected_output_1 = {
    0: {
        "input_shape": (2, 34, 34),
        "pool": [[3, 3], [4, 4]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [1, 2],
        "entry_node": True,
    },
    1: {
        "input_shape": (10, 11, 11),
        "pool": [[1, 1]],
        "rescale_factor": 1.0 / 9,
        "rescale_factors": set(),  # Single factor will be popped from list
        "destination_indices": [3],
        "entry_node": False,
    },
    2: {
        "input_shape": (10, 8, 8),
        "pool": [[1, 1]],
        "rescale_factor": 1.0 / 16,
        "rescale_factors": set(),  # Single factor will be popped from list
        "destination_indices": [3],
        "entry_node": False,
    },
    3: {
        "input_shape": (1, 7, 7),
        "pool": [[1, 1]],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [4],
        "entry_node": False,
    },
    4: {
        "input_shape": (500, 1, 1),
        "pool": [],
        "rescale_factor": 1,
        "rescale_factors": set(),
        "destination_indices": [],
        "entry_node": False,
    },
}
