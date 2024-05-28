# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "a complex network structure" example in https://github.com/synsense/sinabs/issues/181 . """

import torch.nn as nn
from sinabs.layers import IAFSqueeze, SumPool2d
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

nodes_to_dcnnl_map_4 = {
    0: {
        0: {
            'layer': nn.Conv2d(2, 1, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (2, 34, 34), 
            'output_shape': (1, 33, 33)
        }, 
        1: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (1, 33, 33), 
            'output_shape': (1, 33, 33)
        }, 
        'destinations': [1, 2], 
        'conv_rescale_factor': []
    },
    1: {
        2: {
            'layer': nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (1, 33, 33), 
            'output_shape': (1, 32, 32)
        }, 
        4: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (1, 32, 32), 
            'output_shape': (1, 32, 32)
        }, 
        5: {
            'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (1, 32, 32), 
            'output_shape': (1, 16, 16)
        }, 
        'destinations': [3], 
        'conv_rescale_factor': []
    },
    2: {
        3: {
            'layer': nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (1, 33, 33), 
            'output_shape': (1, 32, 32)
        }, 
        7: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (1, 32, 32), 
            'output_shape': (1, 32, 32)
        }, 
        8: {
            'layer': SumPool2d(kernel_size=2, stride=2, ceil_mode=False), 
            'input_shape': (1, 32, 32), 
            'output_shape': (1, 16, 16)
        }, 
        9: {
            'layer': SumPool2d(kernel_size=5, stride=5, ceil_mode=False), 
            'input_shape': (1, 32, 32), 
            'output_shape': (1, 6, 6)
        }, 
        'destinations': [3, 4], 
        'conv_rescale_factor': []
    },
    3: {
        11: {
            'layer': nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (1, 16, 16), 
            'output_shape': (1, 15, 15)
        },
        12: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (1, 15, 15), 
            'output_shape': (1, 15, 15)
        },
        13: {
            'layer': SumPool2d(kernel_size=3, stride=3, ceil_mode=False), 
            'input_shape': (1, 15, 15), 
            'output_shape': (1, 5, 5)
        }, 
        'destinations': [5], 
        'conv_rescale_factor': []
    },
    4: {
        10: {
            'layer': nn.Conv2d(1, 1, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (1, 6, 6), 
            'output_shape': (1, 5, 5)
        }, 
        15: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (1, 5, 5), 
            'output_shape': (1, 5, 5)
        }, 
        'destinations': [5], 
        'conv_rescale_factor': []
    },
    5: {
        16: {
            'layer': nn.Linear(in_features=25, out_features=10, bias=False), 
            'input_shape': (1, 5, 5), 
            'output_shape': (10,)
        },
        17: {
            'layer': IAFSqueeze(batch_size=2, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (10,), 
            'output_shape': (10,)
        }, 
        'destinations': [], 
        'conv_rescale_factor': []
    }
}

sinabs_edges_4 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (4, 5),
    (5, 11),
    (3, 7),
    (7, 8),
    (7, 9),
    (8, 11),
    (9, 10),
    (11, 12),
    (12, 13),
    (13, 16),
    (15, 16),
    (10, 15),
    (16, 17),
]

expected_output_4 = {
    0: {
        'dpcnnl_index': 0,
        'conv_node_id': 0,
        'conv_in_shape': (2, 34, 34),
        'conv_out_shape': (1, 33, 33),
        'spk_node_id': 1,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [1, 2],
        'nodes_destinations': {1: [2, 3]},
        'entry_point': True,
    },
    1: {
        'dpcnnl_index': 1,
        'conv_node_id': 2,
        'conv_in_shape': (1, 33, 33),
        'conv_out_shape': (1, 32, 32),
        'spk_node_id': 4,
        'pool_node_id': [5],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [3],
        'nodes_destinations': {5: [11]},
        'entry_point': False,
    },
    2: {
        'dpcnnl_index': 2,
        'conv_node_id': 3,
        'conv_in_shape': (1, 33, 33),
        'conv_out_shape': (1, 32, 32),
        'spk_node_id': 7,
        'pool_node_id': [8, 9],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [3, 4],
        'nodes_destinations': {8: [11], 9: [10]},
        'entry_point': False,
    },
    3: {
        'dpcnnl_index': 3,
        'conv_node_id': 11,
        'conv_in_shape': (1, 16, 16),
        'conv_out_shape': (1, 15, 15),
        'spk_node_id': 12,
        'pool_node_id': [13],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [5],
        'nodes_destinations': {13: [16]},
        'entry_point': False,
    },
    4: {
        'dpcnnl_index': 4,
        'conv_node_id': 10,
        'conv_in_shape': (1, 6, 6),
        'conv_out_shape': (1, 5, 5),
        'spk_node_id': 15,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [5],
        'nodes_destinations': {15: [16]},
        'entry_point': False,
    },
    5: {
        'dpcnnl_index': 5,
        'conv_node_id': 16,
        'conv_in_shape': (1, 5, 5),
        'conv_out_shape': (10, 1, 1),
        'spk_node_id': 17,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [],
        'nodes_destinations': {},
        'entry_point': False,
    },
}