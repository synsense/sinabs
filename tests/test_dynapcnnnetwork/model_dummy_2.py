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
input_shape = (channels, height, width)

snn = SNN(batch_size)

expected_output = {
    'dcnnl_edges': [
        (0, 1),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 6),
        (4, 5),
        (6, 5),
        ('input', 0),
    ],
    'merge_points': {5: {'sources': (4, 6), 'merge': Merge()}},
    'topological_order': [0, 1, 2, 3, 4, 6, 5],
    'output_shape': torch.Size([8, 10, 1, 1]),
    'entry_point': [0],
}