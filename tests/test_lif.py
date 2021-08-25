import torch
from sinabs.layers import LIF, LIFSqueeze


def test_lif():
    batch_size = 10
    time_steps = 100
    current_input = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = LIF(tau_mem=1., threshold=1.)
    spike_output = layer(current_input)

    assert current_input.shape == spike_output.shape

def test_lif_squeezed():
    batch_size = 10
    time_steps = 100
    current_input = torch.rand(batch_size*time_steps, 2, 7, 7)
    layer = LIFSqueeze(tau_mem=1, threshold=1, batch_size=batch_size)
    spike_output = layer(current_input)
    
    assert current_input.shape == spike_output.shape

def test_lif_backward():
    current_input = torch.tensor([[-5.5, -3.23, 2.3, 0.1]], requires_grad=True)
    layer = LIF(tau_mem=1., threshold=1.)
    output_spikes = layer(current_input)
    z = 2 * output_spikes.sum()
    z.backward()
    grad = output_spikes.grad

#     import ipdb; ipdb.set_trace()
    assert (grad == torch.tensor([0.0, 0.0, 2.0, 2.0])).all()
