# implementing "a network with a merge and a split" in https://github.com/synsense/sinabs/issues/181

import torch.nn as nn
from sinabs.layers import IAFSqueeze, SumPool2d
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

nodes_to_dcnnl_map_2 = {
    0: {
        0: {'layer': nn.Conv2d(2, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (2, 34, 34), 
            'output_shape': (4, 33, 33)
        }, 
        1: {'layer': IAFSqueeze(batch_size=8, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 33, 33), 
            'output_shape': (4, 33, 33)
        }, 
        'destinations': [1], 
        'conv_rescale_factor': []
        },
    1: {
        2: {'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 33, 33), 
            'output_shape': (4, 32, 32)
        },
        3: {'layer': IAFSqueeze(batch_size=8, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 32, 32), 
            'output_shape': (4, 32, 32)
        },
        4: {'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (4, 32, 32), 
            'output_shape': (4, 16, 16)
        }, 
        'destinations': [2, 3], 
        'conv_rescale_factor': []
    },
    2: {
        5: {'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 16, 16), 
            'output_shape': (4, 15, 15)
        }, 
        7: {'layer': IAFSqueeze(batch_size=8, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 15, 15)
        }, 
        8: {'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 7, 7)
        }, 
        'destinations': [4], 
        'conv_rescale_factor': []
    },
    3: {
        6: {'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 16, 16), 
            'output_shape': (4, 15, 15)
        }, 
        11: {'layer': IAFSqueeze(batch_size=8, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 15, 15)
        }, 
        12: {'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (4, 15, 15), 
            'output_shape': (4, 7, 7)
        }, 
        'destinations': [6], 
        'conv_rescale_factor': []
    },
    4: {
        9: {'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 7, 7), 
            'output_shape': (4, 6, 6)
        }, 
        10: {'layer': IAFSqueeze(batch_size=8, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 6, 6), 
            'output_shape': (4, 6, 6)
        }, 
        'destinations': [5], 
        'conv_rescale_factor': []
    },
    5 :{
        15: {'layer': nn.Linear(in_features=144, out_features=10, bias=False), 
            'input_shape': (4, 6, 6), 
            'output_shape': (10,)
        }, 
        16: {'layer': IAFSqueeze(batch_size=8, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (10,), 
            'output_shape': (10,)
        }, 
        'destinations': [],
        'conv_rescale_factor': []
    },
    6: {
        13: {'layer': nn.Conv2d(4, 4, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (4, 7, 7), 
            'output_shape': (4, 6, 6)
        },
        14: {'layer': IAFSqueeze(batch_size=8, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (4, 6, 6), 
            'output_shape': (4, 6, 6)
        }, 
        'destinations': [5], 
        'conv_rescale_factor': []
        }
}

sinabs_edges_2 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    (4, 6),
    (5, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 15),
    (6, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (14, 15),
    (15, 16),
]

expected_output_2 = {
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
        'dynapcnnlayer_destination': [2, 3],
        'nodes_destinations': {4: [5, 6]},
        'entry_point': False,
    },
    2: {
        'dpcnnl_index': 2,
        'conv_node_id': 5,
        'conv_in_shape': (4, 16, 16),
        'conv_out_shape': (4, 15, 15),
        'spk_node_id': 7,
        'pool_node_id': [8],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [4],
        'nodes_destinations': {8: [9]},
        'entry_point': False,
    },
    3: {
        'dpcnnl_index': 3,
        'conv_node_id': 6,
        'conv_in_shape': (4, 16, 16),
        'conv_out_shape': (4, 15, 15),
        'spk_node_id': 11,
        'pool_node_id': [12],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [6],
        'nodes_destinations': {12: [13]},
        'entry_point': False,
    },
    4: {
        'dpcnnl_index': 4,
        'conv_node_id': 9,
        'conv_in_shape': (4, 7, 7),
        'conv_out_shape': (4, 6, 6),
        'spk_node_id': 10,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [5],
        'nodes_destinations': {10: [15]},
        'entry_point': False,
    },
    5: {
        'dpcnnl_index': 5,
        'conv_node_id': 15,
        'conv_in_shape': (4, 6, 6),
        'conv_out_shape': (10, 1, 1),
        'spk_node_id': 16,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [],
        'nodes_destinations': {},
        'entry_point': False,
    },
    6: {
        'dpcnnl_index': 6,
        'conv_node_id': 13,
        'conv_in_shape': (4, 7, 7),
        'conv_out_shape': (4, 6, 6),
        'spk_node_id': 14,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [5],
        'nodes_destinations': {14: [15]},
        'entry_point': False,
    },
}