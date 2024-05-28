# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com
# implementing "a complex network structure" example in https://github.com/synsense/sinabs/issues/181 . """

import torch
import torch.nn as nn
from sinabs.layers import IAFSqueeze, SumPool2d, Merge
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

class SNN(nn.Module):
    def __init__(self, batch_size) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(2, 1, 2, 1, bias=False)
        self.iaf1 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())

        self.conv2 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf2 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool2 = SumPool2d(2,2)

        self.conv3 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf3 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool3 = SumPool2d(2,2)
        self.pool3a = SumPool2d(5,5)

        self.conv4 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf4 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())
        self.pool4 = SumPool2d(3,3)

        self.flat1 = nn.Flatten()
        self.flat2 = nn.Flatten()

        self.conv5 = nn.Conv2d(1, 1, 2, 1, bias=False)
        self.iaf5 = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())

        self.fc2 = nn.Linear(25, 10, bias=False)
        self.iaf2_fc = IAFSqueeze(batch_size=batch_size, min_v_mem=-1.0, spike_threshold=1.0, surrogate_grad_fn=PeriodicExponential())

        # -- merges --
        self.merge1 = Merge()
        self.merge2 = Merge()

    def forward(self, x):
        # conv 1 - A/0
        con1_out = self.conv1(x)
        iaf1_out = self.iaf1(con1_out)

        # conv 2 - B/1
        conv2_out = self.conv2(iaf1_out)
        iaf2_out = self.iaf2(conv2_out)
        pool2_out = self.pool2(iaf2_out)

        # conv 3 - C/2
        conv3_out = self.conv3(iaf1_out)
        iaf3_out = self.iaf3(conv3_out)
        pool3_out = self.pool3(iaf3_out)
        pool3a_out = self.pool3a(iaf3_out)

        # conv 4 - D/3
        merge1_out = self.merge1(pool2_out, pool3_out)
        conv4_out = self.conv4(merge1_out)
        iaf4_out = self.iaf4(conv4_out)
        pool4_out = self.pool4(iaf4_out)
        flat1_out = self.flat1(pool4_out)
        
        # conv 5 - E/4
        conv5_out = self.conv5(pool3a_out)
        iaf5_out = self.iaf5(conv5_out)
        flat2_out = self.flat2(iaf5_out)

        # fc 2 - F/5
        merge2_out = self.merge2(flat2_out, flat1_out)
        
        fc2_out = self.fc2(merge2_out)
        iaf2_fc_out = self.iaf2_fc(fc2_out)

        return iaf2_fc_out
    
channels = 2
height = 34
width = 34
batch_size = 2
input_shape = (channels, height, width)

snn = SNN(batch_size)

expected_output = {
    'dcnnl_edges': [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 5),
        (4, 5),
        ('input', 0),
    ],
    'merge_points': {
        3: {'sources': (1, 2), 'merge': Merge()},
        5: {'sources': (3, 4), 'merge': Merge()},
    },
    'topological_order': [0, 1, 2, 3, 4, 5],
    'output_shape': torch.Size([2, 10, 1, 1]),
    'entry_point': [0],
}