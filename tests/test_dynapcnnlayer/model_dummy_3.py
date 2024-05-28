# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "two networks with merging outputs" in https://github.com/synsense/sinabs/issues/181

import torch.nn as nn
from sinabs.layers import IAFSqueeze, SumPool2d
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

nodes_to_dcnnl_map_3 = {
    0: {
        0: {
            'layer': nn.Conv2d(2, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (2, 34, 34), 
            'output_shape': (4, 33, 33)
        }, 
        1: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 33, 33), 
            'output_shape': (4, 33, 33)
        }, 
        'destinations': [1], 
        'conv_rescale_factor': []
    },
    1: {
        2: {
            'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 33, 33), 
            'output_shape': (4, 32, 32)
        }, 
        3: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 32, 32), 
            'output_shape': (4, 32, 32)
        }, 
        4: {
            'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (4, 32, 32), 
            'output_shape': (4, 16, 16)
        }, 
        'destinations': [2], 
        'conv_rescale_factor': []
    },
    2: {
        5: {
            'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 16, 16), 
            'output_shape': (4, 15, 15)}, 
        6: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 15, 15)
        }, 
        7: {
            'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 7, 7)
        }, 
        'destinations': [3], 
        'conv_rescale_factor': []
    },
    3: {
        17: {
            'layer': nn.Linear(in_features=196, out_features=100, bias=False), 
            'input_shape': (4, 7, 7), 
            'output_shape': (100,)
        }, 
        18: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (100,), 
            'output_shape': (100,)
        }, 
        'destinations': [7], 
        'conv_rescale_factor': []
    },
    4: {
        8: {
            'layer': nn.Conv2d(2, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (2, 34, 34), 
            'output_shape': (4, 33, 33)}, 
        9: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 33, 33), 
            'output_shape': (4, 33, 33)
        }, 
        'destinations': [5], 
        'conv_rescale_factor': []
    },
    5: {
        10: {
            'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 33, 33), 
            'output_shape': (4, 32, 32)
        },
        11: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 32, 32), 
            'output_shape': (4, 32, 32)
        }, 
        12: {
            'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (4, 32, 32), 
            'output_shape': (4, 16, 16)
        }, 
        'destinations': [6], 
        'conv_rescale_factor': []
    },
    6: {
        13: {
            'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 16, 16), 
            'output_shape': (4, 15, 15)
        }, 
        14: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 15, 15)
        }, 
        15: {
            'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 7, 7)
        }, 
        'destinations': [3], 
        'conv_rescale_factor': []
    },
    7: {
        19: {
            'layer': nn.Linear(in_features=100, out_features=100, bias=False), 
            'input_shape': (100,), 
            'output_shape': (100,)
        }, 
        20: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (100,), 
            'output_shape': (100,)
        }, 
        'destinations': [8], 
        'conv_rescale_factor': []
    },
    8: {
        21: {
            'layer': nn.Linear(in_features=100, out_features=10, bias=False), 
            'input_shape': (100,), 
            'output_shape': (10,)
        }, 
        22: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (10,), 
            'output_shape': (10,)
        }, 
        'destinations': [], 
        'conv_rescale_factor': []
    }
}

sinabs_edges_3 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 17),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    (21, 22),
]

expected_output_3 = {
    0: {
        'dpcnnl_index': 0,
        'conv_node_id': 0,
        'conv_in_shape': (2, 34, 34),
        'conv_out_shape': (4, 33, 33),
        'spk_node_id': 1,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [1],
        'nodes_destinations': {1: [2]},
        'entry_point': True,
    },
    1: {
        'dpcnnl_index': 1,
        'conv_node_id': 2,
        'conv_in_shape': (4, 33, 33),
        'conv_out_shape': (4, 32, 32),
        'spk_node_id': 3,
        'pool_node_id': [4],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [2],
        'nodes_destinations': {4: [5]},
        'entry_point': False,
    },
    2: {
        'dpcnnl_index': 2,
        'conv_node_id': 5,
        'conv_in_shape': (4, 16, 16),
        'conv_out_shape': (4, 15, 15),
        'spk_node_id': 6,
        'pool_node_id': [7],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [3],
        'nodes_destinations': {7: [17]},
        'entry_point': False,
    },
    3: {
        'dpcnnl_index': 3,
        'conv_node_id': 17,
        'conv_in_shape': (4, 7, 7),
        'conv_out_shape': (100, 1, 1),
        'spk_node_id': 18,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [7],
        'nodes_destinations': {18: [19]},
        'entry_point': False,
    },
    4: {
        'dpcnnl_index': 4,
        'conv_node_id': 8,
        'conv_in_shape': (2, 34, 34),
        'conv_out_shape': (4, 33, 33),
        'spk_node_id': 9,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [5],
        'nodes_destinations': {9: [10]},
        'entry_point': True,
    },
    5: {
        'dpcnnl_index': 5,
        'conv_node_id': 10,
        'conv_in_shape': (4, 33, 33),
        'conv_out_shape': (4, 32, 32),
        'spk_node_id': 11,
        'pool_node_id': [12],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [6],
        'nodes_destinations': {12: [13]},
        'entry_point': False,
    },
    6: {
        'dpcnnl_index': 6,
        'conv_node_id': 13,
        'conv_in_shape': (4, 16, 16),
        'conv_out_shape': (4, 15, 15),
        'spk_node_id': 14,
        'pool_node_id': [15],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [3],
        'nodes_destinations': {15: [17]},
        'entry_point': False,
    },
    7: {
        'dpcnnl_index': 7,
        'conv_node_id': 19,
        'conv_in_shape': (100, 1, 1),
        'conv_out_shape': (100, 1, 1),
        'spk_node_id': 20,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [8],
        'nodes_destinations': {20: [21]},
        'entry_point': False,
    },
    8: {
        'dpcnnl_index': 8,
        'conv_node_id': 21,
        'conv_in_shape': (100, 1, 1),
        'conv_out_shape': (10, 1, 1),
        'spk_node_id': 22,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [],
        'nodes_destinations': {},
        'entry_point': False,
    },
}