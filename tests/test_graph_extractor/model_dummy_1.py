# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "a network with residual connections" example in https://github.com/synsense/sinabs/issues/181

import torch
import torch.nn as nn
from sinabs.layers import Merge, IAFSqueeze
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

class SNN(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 10, 2, 1, bias=False) # node 0
        self.iaf1 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())            # node 1
        self.pool1 = nn.AvgPool2d(3,3)                  # node 2
        self.pool1a = nn.AvgPool2d(4,4)                 # node 3

        self.conv2 = nn.Conv2d(10, 10, 4, 1, bias=False)# node 4
        self.iaf2 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())            # node 6

        self.conv3 = nn.Conv2d(10, 1, 2, 1, bias=False) # node 8
        self.iaf3 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())            # node 9

        self.flat = nn.Flatten()

        self.fc1 = nn.Linear(49, 500, bias=False)       # node 10
        self.iaf4 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())            # node 11
        
        self.fc2 = nn.Linear(500, 10, bias=False)       # node 12
        self.iaf5 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())            # node 13

        self.adder = Merge()

    def forward(self, x):
        
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)
        pool1_out = self.pool1(iaf1_out)
        pool1a_out = self.pool1a(iaf1_out)

        conv2_out = self.conv2(pool1_out)
        iaf2_out = self.iaf2(conv2_out)

        conv3_out = self.conv3(self.adder(pool1a_out, iaf2_out))
        iaf3_out = self.iaf3(conv3_out)

        flat_out = self.flat(iaf3_out)
        
        fc1_out = self.fc1(flat_out)
        iaf4_out = self.iaf4(fc1_out)
        fc2_out = self.fc2(iaf4_out)
        iaf5_out = self.iaf5(fc2_out)

        return iaf5_out
    
channels = 2
height = 34
width = 34
batch_size = 3
input_shape = (batch_size, channels, height, width)

torch.manual_seed(0)
input_dummy = torch.randn(input_shape)

expected_output = {
    'edges': {
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (6, 5),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (5, 7),
    },
    'name_2_indx_map': {
        'conv1': 0,
        'iaf1': 1,
        'pool1': 2,
        'pool1a': 3,
        'conv2': 4,
        'adder': 5,
        'iaf2': 6,
        'conv3': 7,
        'iaf3': 8,
        'flat': 9,
        'fc1': 10,
        'iaf4': 11,
        'fc2': 12,
        'iaf5': 13,
    },
    'entry_nodes': {0},
    'nodes_io_shapes': {
        0: {'input': torch.Size([3, 2, 34, 34]), 'output': torch.Size([3, 10, 33, 33])},
        1: {'input': torch.Size([3, 10, 33, 33]), 'output': torch.Size([3, 10, 33, 33])},
        2: {'input': torch.Size([3, 10, 33, 33]), 'output': torch.Size([3, 10, 11, 11])},
        3: {'input': torch.Size([3, 10, 33, 33]), 'output': torch.Size([3, 10, 8, 8])},
        4: {'input': torch.Size([3, 10, 11, 11]), 'output': torch.Size([3, 10, 8, 8])},
        6: {'input': torch.Size([3, 10, 8, 8]), 'output': torch.Size([3, 10, 8, 8])},
        5: {'input': torch.Size([3, 10, 8, 8]), 'output': torch.Size([3, 10, 8, 8])},
        7: {'input': torch.Size([3, 10, 8, 8]), 'output': torch.Size([3, 1, 7, 7])},
        8: {'input': torch.Size([3, 1, 7, 7]), 'output': torch.Size([3, 1, 7, 7])},
        9: {'input': torch.Size([3, 1, 7, 7]), 'output': torch.Size([3, 49])},
        10: {'input': torch.Size([3, 49]), 'output': torch.Size([3, 500])},
        11: {'input': torch.Size([3, 500]), 'output': torch.Size([3, 500])},
        12: {'input': torch.Size([3, 500]), 'output': torch.Size([3, 10])},
        13: {'input': torch.Size([3, 10]), 'output': torch.Size([3, 10])},
    },
}

snn = SNN(batch_size)
