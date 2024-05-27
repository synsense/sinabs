import torch.nn as nn
from sinabs.layers import IAFSqueeze, SumPool2d
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

nodes_to_dcnnl_map_1 = {
    0: {
        0: {
            'layer': nn.Conv2d(2, 10, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (2, 34, 34), 
            'output_shape': (10, 33, 33)
            }, 
        1: {
            'layer': IAFSqueeze(batch_size=3, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (10, 33, 33), 
            'output_shape': (10, 33, 33)
            }, 
        2: {
            'layer': nn.AvgPool2d(kernel_size=3, stride=3, padding=0), 
            'input_shape': (10, 33, 33), 
            'output_shape': (10, 11, 11)
            }, 
        3: {
            'layer': nn.AvgPool2d(kernel_size=4, stride=4, padding=0), 
            'input_shape': (10, 33, 33), 
            'output_shape': (10, 8, 8)
            }, 
        'destinations': [1, 2], 
        'conv_rescale_factor': []
        },
    1: {
        4: {
            'layer': nn.Conv2d(10, 10, kernel_size=(4, 4), stride=(1, 1), bias=False), 
            'input_shape': (10, 11, 11), 
            'output_shape': (10, 8, 8)
            }, 
        6: {
            'layer': IAFSqueeze(batch_size=3, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (10, 8, 8), 
            'output_shape': (10, 8, 8)
            }, 
            'destinations': [2], 
            'conv_rescale_factor': [9]
        },
    2: {
        7: {
            'layer': nn.Conv2d(10, 1, kernel_size=(2, 2), stride=(1, 1), bias=False), 
            'input_shape': (10, 8, 8), 
            'output_shape': (1, 7, 7)
            }, 
        8: {
            'layer': IAFSqueeze(batch_size=3, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (1, 7, 7),
            'output_shape': (1, 7, 7)
            }, 
        'destinations': [3], 
        'conv_rescale_factor': [16]
        },
    3: {
        9: {
            'layer': nn.Linear(in_features=49, out_features=500, bias=False), 
            'input_shape': (1, 7, 7), 
            'output_shape': (500,)
            }, 
        10: {
            'layer': IAFSqueeze(batch_size=3, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (500,), 
            'output_shape': (500,)
            }, 
        'destinations': [4], 
        'conv_rescale_factor': []
        },
    4: {
        11: {
            'layer': nn.Linear(in_features=500, out_features=10, bias=False), 
            'input_shape': (500,), 
            'output_shape': (10,)
            }, 
        12: {
            'layer': IAFSqueeze(batch_size=3, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential()), 
            'input_shape': (10,), 
            'output_shape': (10,)
            }, 
        'destinations': [], 
        'conv_rescale_factor': []
        }   
}

sinabs_edges_1 = [
    (0, 1),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 7),
    (4, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (10, 11),
    (11, 12),
]

expected_output_1 = {
    0: {
        'dpcnnl_index': 0,
        'conv_node_id': 0,
        'conv_in_shape': (2, 34, 34),
        'conv_out_shape': (10, 33, 33),
        'spk_node_id': 1,
        'pool_node_id': [2, 3],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [1, 2],
        'nodes_destinations': {2: [4], 3: [7]},
    },
    1: {
        'dpcnnl_index': 1,
        'conv_node_id': 4,
        'conv_in_shape': (10, 11, 11),
        'conv_out_shape': (10, 8, 8),
        'spk_node_id': 6,
        'pool_node_id': [],
        'conv_rescaling_factor': 4.5,
        'dynapcnnlayer_destination': [2],
        'nodes_destinations': {6: [7]},
    },
    2: {
        'dpcnnl_index': 2,
        'conv_node_id': 7,
        'conv_in_shape': (10, 8, 8),
        'conv_out_shape': (1, 7, 7),
        'spk_node_id': 8,
        'pool_node_id': [],
        'conv_rescaling_factor': 8.0,
        'dynapcnnlayer_destination': [3],
        'nodes_destinations': {8: [9]},
    },
    3: {
        'dpcnnl_index': 3,
        'conv_node_id': 9,
        'conv_in_shape': (1, 7, 7),
        'conv_out_shape': (500, 1, 1),
        'spk_node_id': 10,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [4],
        'nodes_destinations': {10: [11]},
    },
    4: {
        'dpcnnl_index': 4,
        'conv_node_id': 11,
        'conv_in_shape': (500, 1, 1),
        'conv_out_shape': (10, 1, 1),
        'spk_node_id': 12,
        'pool_node_id': [],
        'conv_rescaling_factor': None,
        'dynapcnnlayer_destination': [],
        'nodes_destinations': {},
    },
}

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
    },
}

args_DynapcnnLayer = [
    (nodes_to_dcnnl_map_1, 0, sinabs_edges_1, expected_output_1), 
    (nodes_to_dcnnl_map_1, 1, sinabs_edges_1, expected_output_1), 
    (nodes_to_dcnnl_map_1, 2, sinabs_edges_1, expected_output_1), 
    (nodes_to_dcnnl_map_1, 3, sinabs_edges_1, expected_output_1), 
    (nodes_to_dcnnl_map_1, 4, sinabs_edges_1, expected_output_1), 
    (nodes_to_dcnnl_map_2, 0, sinabs_edges_2, expected_output_2), 
    (nodes_to_dcnnl_map_2, 1, sinabs_edges_2, expected_output_2), 
    (nodes_to_dcnnl_map_2, 2, sinabs_edges_2, expected_output_2), 
    (nodes_to_dcnnl_map_2, 3, sinabs_edges_2, expected_output_2), 
    (nodes_to_dcnnl_map_2, 4, sinabs_edges_2, expected_output_2), 
    (nodes_to_dcnnl_map_2, 5, sinabs_edges_2, expected_output_2), 
    (nodes_to_dcnnl_map_2, 6, sinabs_edges_2, expected_output_2), 
]