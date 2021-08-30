import torch
from sinabs.layers import ALIF, ALIFSqueeze


def test_alif():
    batch_size = 10
    time_steps = 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = ALIF(tau_mem=1, threshold=1, tau_threshold=10, threshold_adaptation=0.3)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_squeezed():
    batch_size = 10
    time_steps = 100
    input_current = torch.rand(batch_size*time_steps, 2, 7, 7)
    layer = ALIFSqueeze(tau_mem=1, threshold=1, tau_threshold=1, batch_size=batch_size)
    spike_output = layer(input_current)
    
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_minimum_spike_threshold():
    batch_size = 10
    time_steps = 100
    threshold = 10
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    layer = ALIF(tau_mem=100, threshold=threshold, tau_threshold=100)
    spike_output = layer(input_current)

    assert (layer.threshold >= threshold).all(), "Spike thresholds should not drop below initital threshold."

def test_alif_spike_threshold_decay():
    batch_size = 10
    time_steps = 100
    threshold = 1
    input_current = torch.ones(batch_size, time_steps, 2, 7, 7)
    input_current[:,1:,:,:] = 0 # only inject current in the first time step
    layer = ALIF(tau_mem=100, tau_threshold=time_steps, threshold=threshold, threshold_adaptation=1)
    spike_output = layer(input_current)

    assert (layer.threshold >= threshold).all()
    # decay only starts after 2 time steps: current integration and adaption
    threshold_decay = torch.exp(torch.tensor(-(time_steps-2)/time_steps))
    # account for rounding errors with .isclose()
    assert torch.isclose(layer.threshold-threshold, threshold_decay, atol=1e-08).all(), "Neuron spike thresholds do not seems to decay correctly."
