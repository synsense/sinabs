import torch
from sinabs.layers import LIF, LIFSqueeze
import numpy as np


def test_lif():
    batch_size = 10
    time_steps = 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = LIF(tau_mem=1, threshold=1)
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

def test_lif_input_integration():
    batch_size = 10
    time_steps = 100
    threshold = 10
    input_current = torch.ones(batch_size, time_steps, 2, 7, 7) * threshold
    input_current[:,1:,:,:] = 0 # only inject current in the first time step
    layer = LIF(tau_mem=time_steps, threshold=threshold)
    spike_output = layer(input_current)

    assert spike_output.sum() == np.product(input_current.shape) / time_steps, "Every neuron should spike exactly once."
    assert spike_output[:,1].sum() == spike_output.sum(), "First output time step should contain all the spikes."

def test_lif_membrane_decay():
    batch_size = 10
    time_steps = 100
    input_current = torch.ones(batch_size, time_steps, 2, 7, 7)
    input_current[:,1:,:,:] = 0 # only inject current in the first time step
    layer = LIF(tau_mem=time_steps, threshold=10)
    spike_output = layer(input_current)

    # first time step is not decayed
    membrane_decay = torch.exp(torch.tensor(-(time_steps-1)/time_steps))
    # account for rounding errors with .isclose()
    assert torch.isclose(layer.state, membrane_decay, atol=1e-08).all(), "Neuron membrane potentials do not seems to decay correctly."
