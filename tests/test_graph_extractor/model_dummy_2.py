# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "a network with a merge and a split" in https://github.com/synsense/sinabs/issues/181

import torch
import torch.nn as nn
from sinabs.layers import Merge, IAFSqueeze, SumPool2d
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

class SNN(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()
        # -- graph node A --
        self.conv_A = nn.Conv2d(2, 4, 2, 1, bias=False)
        self.iaf_A = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        # -- graph node B --
        self.conv_B = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf2_B = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool_B = SumPool2d(2,2)
        # -- graph node C --
        self.conv_C = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_C = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool_C = SumPool2d(2,2)
        # -- graph node D --
        self.conv_D = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_D = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        # -- graph node E --
        self.conv_E = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf3_E = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool_E = SumPool2d(2,2)
        # -- graph node F --
        self.conv_F = nn.Conv2d(4, 4, 2, 1, bias=False)
        self.iaf_F = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        # -- graph node G --
        self.fc3 = nn.Linear(144, 10, bias=False)
        self.iaf3_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())

        # -- merges --
        self.merge1 = Merge()

        # -- falts --
        self.flat_D = nn.Flatten()
        self.flat_F = nn.Flatten()

    def forward(self, x):
        # conv 1 - A/0
        convA_out = self.conv_A(x)                          # node 0
        iaf_A_out = self.iaf_A(convA_out)                   # node 1

        # conv 2 - B/1
        conv_B_out = self.conv_B(iaf_A_out)                 # node 2
        iaf_B_out = self.iaf2_B(conv_B_out)                 # node 3
        pool_B_out = self.pool_B(iaf_B_out)                 # node 4

        # conv 3 - C/2
        conv_C_out = self.conv_C(pool_B_out)                # node 5
        iaf_C_out = self.iaf_C(conv_C_out)                  # node 7
        pool_C_out = self.pool_C(iaf_C_out)                 # node 8

        # conv 4 - D/4
        conv_D_out = self.conv_D(pool_C_out)                # node 9
        iaf_D_out = self.iaf_D(conv_D_out)                  # node 10
        
        # fc 1 - E/3
        conv_E_out = self.conv_E(pool_B_out)                # node 6
        iaf3_E_out = self.iaf3_E(conv_E_out)                # node 12
        pool_E_out = self.pool_E(iaf3_E_out)                # node 13

        # fc 2 - F/6
        conv_F_out = self.conv_F(pool_E_out)                # node 14
        iaf_F_out = self.iaf_F(conv_F_out)                  # node 15
        
        # fc 2 - G/5
        flat_D_out = self.flat_D(iaf_D_out)                 # node 11
        flat_F_out = self.flat_F(iaf_F_out)                 # node 16
        
        merge1_out = self.merge1(flat_D_out, flat_F_out)    # node 19
        fc3_out = self.fc3(merge1_out)                      # node 17
        iaf3_fc_out = self.iaf3_fc(fc3_out)                 # node 18

        return iaf3_fc_out
    
channels = 2
height = 34
width = 34
batch_size = 8
input_shape = (batch_size, channels, height, width)

torch.manual_seed(0)
input_dummy = torch.randn(input_shape)

expected_output = {
    'edges_list': [
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
        (10, 11),
        (6, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (17, 18),
        (19, 17),
        (11, 19),
        (16, 19),
    ],
    'name_2_indx_map': {
        'conv_A': 0,
        'iaf_A': 1,
        'conv_B': 2,
        'iaf2_B': 3,
        'pool_B': 4,
        'conv_C': 5,
        'conv_E': 6,
        'iaf_C': 7,
        'pool_C': 8,
        'conv_D': 9,
        'iaf_D': 10,
        'flat_D': 11,
        'iaf3_E': 12,
        'pool_E': 13,
        'conv_F': 14,
        'iaf_F': 15,
        'flat_F': 16,
        'fc3': 17,
        'iaf3_fc': 18,
        'merge1': 19,
    },
    'entry_nodes': [0],
    'nodes_io_shapes': {
        0: {'input': torch.Size([8, 2, 34, 34]), 'output': torch.Size([8, 4, 33, 33])},
        1: {'input': torch.Size([8, 4, 33, 33]), 'output': torch.Size([8, 4, 33, 33])},
        2: {'input': torch.Size([8, 4, 33, 33]), 'output': torch.Size([8, 4, 32, 32])},
        3: {'input': torch.Size([8, 4, 32, 32]), 'output': torch.Size([8, 4, 32, 32])},
        4: {'input': torch.Size([8, 4, 32, 32]), 'output': torch.Size([8, 4, 16, 16])},
        5: {'input': torch.Size([8, 4, 16, 16]), 'output': torch.Size([8, 4, 15, 15])},
        6: {'input': torch.Size([8, 4, 16, 16]), 'output': torch.Size([8, 4, 15, 15])},
        7: {'input': torch.Size([8, 4, 15, 15]), 'output': torch.Size([8, 4, 15, 15])},
        12: {'input': torch.Size([8, 4, 15, 15]), 'output': torch.Size([8, 4, 15, 15])},
        8: {'input': torch.Size([8, 4, 15, 15]), 'output': torch.Size([8, 4, 7, 7])},
        13: {'input': torch.Size([8, 4, 15, 15]), 'output': torch.Size([8, 4, 7, 7])},
        9: {'input': torch.Size([8, 4, 7, 7]), 'output': torch.Size([8, 4, 6, 6])},
        14: {'input': torch.Size([8, 4, 7, 7]), 'output': torch.Size([8, 4, 6, 6])},
        10: {'input': torch.Size([8, 4, 6, 6]), 'output': torch.Size([8, 4, 6, 6])},
        15: {'input': torch.Size([8, 4, 6, 6]), 'output': torch.Size([8, 4, 6, 6])},
        11: {'input': torch.Size([8, 4, 6, 6]), 'output': torch.Size([8, 144])},
        16: {'input': torch.Size([8, 4, 6, 6]), 'output': torch.Size([8, 144])},
        19: {'input': torch.Size([8, 144]), 'output': torch.Size([8, 144])},
        17: {'input': torch.Size([8, 144]), 'output': torch.Size([8, 10])},
        18: {'input': torch.Size([8, 10]), 'output': torch.Size([8, 10])},
    },
}

snn = SNN(batch_size)