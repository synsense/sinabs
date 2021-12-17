import torch
import torch.nn as nn
from sinabs.layers import ALIF, ALIFRecurrent
import pytest
import numpy as np


def test_alif_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)  / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, adapt_scale=1.8)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_minimum_spike_threshold():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem,)
    spike_output = layer(input_current)

    assert (layer.b0 + layer.b >= 1.).all(), "Spike thresholds should not drop below initital threshold."
    assert spike_output.sum() > 0
    assert (layer.b0 + layer.b > 1.).all(), "Spike thresholds should be above default threshold if any spikes occured."

def test_alif_spike_threshold_decay():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    input_current[:,0] = 1 / (1-alpha) # only inject current in the first time step and make it spike
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, adapt_scale=1 / (1-alpha))
    spike_output = layer(input_current)
    
    layer2 = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, adapt_scale=1 / (1-alpha))

    assert (layer.threshold > 1).all()
    assert spike_output.sum() == torch.prod(torch.as_tensor(input_current.size())) / time_steps, "All neurons should spike exactly once."
    # decay only starts after 1 time step
    threshold_decay = alpha ** (time_steps-1)
    # account for rounding errors with .isclose()
    assert torch.isclose(layer.threshold-1, threshold_decay, atol=1e-08).all(), "Neuron spike thresholds do not seems to decay correctly."

def test_alif_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    tau_syn = torch.tensor(10.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, tau_syn=tau_syn, tau_adapt=tau_mem)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_alif_train_alphas():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, train_alphas=True, tau_adapt=tau_mem)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    
def test_alif_train_alphas_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    tau_syn = torch.tensor(10.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, tau_syn=tau_syn, tau_adapt=tau_mem, train_alphas=True)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
     
def test_alif_recurrent():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5 / (1-alpha)
    
    rec_connect = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connect[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = ALIFRecurrent(tau_mem=tau_mem, tau_adapt=tau_mem, rec_connect=rec_connect)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alif_device_movement():
    batch_size, time_steps = 10, 100
    tau_mem = torch.as_tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)  / (1-alpha)
    layer = ALIF(tau_mem=tau_mem, tau_adapt=tau_mem, adapt_scale=1.8)
    
    layer = layer.to("cuda")
    layer(input_current.to("cuda"))