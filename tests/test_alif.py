import torch
import torch.nn as nn
from sinabs.layers import ALIF, ALIFSqueeze, ALIFRecurrent, ALIFRecurrentSqueeze
import pytest
import numpy as np


def test_alif_basic():
    batch_size = 10
    time_steps = 30
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)  / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, threshold=0.1, tau_adapt=tau_mem, adapt_scale=1.8)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_squeezed():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size*time_steps, 2, 7, 7) / (1-alpha)
    layer = ALIFSqueeze(tau_mem=tau_mem, threshold=1, tau_adapt=tau_mem, batch_size=batch_size)
    spike_output = layer(input_current)
    
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_minimum_spike_threshold():
    batch_size = 10
    time_steps = 100
    threshold = 10
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    layer = ALIF(tau_mem=tau_mem, threshold=threshold, tau_adapt=tau_mem,)
    spike_output = layer(input_current)

    assert (layer.b_0 + layer.adapt_scale + layer.b >= threshold).all(), "Spike thresholds should not drop below initital threshold."

# def test_alif_spike_threshold_decay():
#     batch_size = 10
#     time_steps = 100
#     threshold = 1
#     alpha = torch.tensor(0.995)
#     adapt_scale = float(1/(1-alpha))
#     input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
#     input_current[:,0,:,:] = 1 / (1-alpha)# only inject current in the first time step
#     layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, threshold=threshold, adapt_scale=adapt_scale)
#     spike_output = layer(input_current)

#     assert (layer.threshold > threshold).all()
#     # decay only starts after 2 time steps: current integration and adaption
#     threshold_decay = alpha ** (time_steps-2)
#     # account for rounding errors with .isclose()
#     assert torch.isclose(layer.threshold-threshold, threshold_decay, atol=1e-08).all(), "Neuron spike thresholds do not seems to decay correctly."

        
def test_alif_recurrent():
    batch_size = 5
    time_steps = 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5 / (1-alpha)
    
    rec_connectivity = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connectivity[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = ALIFRecurrent(tau_mem=tau_mem, tau_adapt=tau_mem, rec_connectivity=rec_connectivity)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_recurrent_squeezed():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_dimensions = (batch_size*time_steps, 2, 7, 7)
    n_neurons = np.product(input_dimensions[1:])
    input_current = torch.rand(*input_dimensions) / (1-alpha)

    rec_connectivity = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connectivity[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = ALIFRecurrentSqueeze(tau_mem=tau_mem, tau_adapt=tau_mem, threshold=1, batch_size=batch_size, rec_connectivity=rec_connectivity)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0