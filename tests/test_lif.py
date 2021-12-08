import torch
import torch.nn as nn
from sinabs.layers import LIF, LIFSqueeze, LIFRecurrent, LIFRecurrentSqueeze
import numpy as np
import pytest


def test_lif_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = LIF(tau_mem=tau_mem)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    
def test_lif_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    tau_syn = torch.tensor(10.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = LIF(tau_mem=tau_mem, tau_syn=tau_syn)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_lif_train_alphas():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = LIF(tau_mem=tau_mem, train_alphas=True)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    
def test_lif_train_alphas_with_current_dynamics():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    tau_syn = torch.tensor(10.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7) / (1-alpha)
    layer = LIF(tau_mem=tau_mem, tau_syn=tau_syn, train_alphas=True)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_lif_input_integration():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    input_current[:,0,:,:] = 1 / (1-alpha)# only inject current in the first time step
    layer = LIF(tau_mem=tau_mem)
    spike_output = layer(input_current)

    assert spike_output.sum() == np.product(input_current.shape) / time_steps, "Every neuron should spike exactly once."
    assert spike_output[:,0].sum() == spike_output.sum(), "First output time step should contain all the spikes."

def test_lif_membrane_decay():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    start_value = 0.75
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    input_current[:,0,:,:] = start_value / (1-alpha) # only inject current in the first time step
    layer = LIF(tau_mem=tau_mem)
    spike_output = layer(input_current)

    # decay only starts after first time step
    membrane_decay = start_value * alpha ** (time_steps-1)
    # account for rounding errors with .isclose()
    assert torch.isclose(layer.v_mem, membrane_decay, atol=1e-08).all(), "Neuron membrane potentials do not seems to decay correctly."

def test_lif_squeezed():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size*time_steps, 2, 7, 7) / (1-alpha)
    layer = LIFSqueeze(tau_mem=tau_mem, batch_size=batch_size)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    
def test_lif_recurrent_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5 / (1-alpha)
    
    rec_connect = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connect[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = LIFRecurrent(tau_mem=tau_mem, rec_connect=rec_connect)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_lif_recurrent_squeezed():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_dimensions = (batch_size*time_steps, 2, 7, 7)
    n_neurons = np.product(input_dimensions[1:])
    input_current = torch.rand(*input_dimensions) / (1-alpha)

    rec_connect = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connect[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = LIFRecurrentSqueeze(tau_mem=tau_mem, batch_size=batch_size, rec_connect=rec_connect)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
