import torch
from sinabs.layers import LIF, LIFSqueeze


def test_lif():
    batch_size = 10
    time_steps = 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = LIF(tau_mem=1., threshold=1.)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_lif_squeezed():
    batch_size = 10
    time_steps = 100
    input_current = torch.rand(batch_size*time_steps, 2, 7, 7)
    layer = LIFSqueeze(tau_mem=1, threshold=1, batch_size=batch_size)
    spike_output = layer(input_current)
    
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
