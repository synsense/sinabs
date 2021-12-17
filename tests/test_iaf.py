import torch
import torch.nn as nn
from sinabs.layers import IAF, IAFSqueeze, IAFRecurrent
import pytest
import numpy as np


def test_iaf_basic():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = IAF()
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_iaf_squeezed():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size*time_steps, 2, 7, 7)
    layer = IAFSqueeze(batch_size=batch_size)
    spike_output = layer(input_current)
    
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
     
def test_iaf_recurrent():
    batch_size, time_steps = 10, 100
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5
    
    rec_connect = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    rec_connect[1].weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5)
    layer = IAFRecurrent(rec_connect=rec_connect)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alif_on_gpu():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = IAF()
    
    layer = layer.to("cuda")
    layer(input_current.to("cuda"))
    
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_alif_recurrent_on_gpu():
    batch_size, time_steps = 10, 100
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions)
    
    rec_connect = nn.Sequential(nn.Flatten(), nn.Linear(n_neurons, n_neurons, bias=False))
    layer = IAFRecurrent(rec_connect=rec_connect)
    
    layer = layer.to("cuda")
    layer(input_current.to("cuda"))