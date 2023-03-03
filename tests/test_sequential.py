import torch
import torch.nn as nn

import sinabs.nn as snn


def test_sequential():
    fc = nn.Linear(10, 20)
    lif = snn.LIF(30.0)

    net = snn.Sequential(fc, lif)

    state1 = dict(
        v_mem=torch.zeros((num_hidden), device=device),
        i_syn=torch.zeros((num_hidden), device=device),
    )
    state2 = dict(
        v_mem=torch.zeros((num_outputs), device=device),
        i_syn=torch.zeros((num_outputs), device=device),
    )

    input_data
    output, state = net(input_data)
    assert True
