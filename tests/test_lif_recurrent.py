import torch
import torch.nn as nn
from sinabs.layers import LIFRecurrent, LIFRecurrentSqueeze
import numpy as np
import pytest


def test_lif_recurrent():
    batch_size = 5
    time_steps = 100
    alpha = torch.tensor(0.99)
    input_dimensions = (batch_size, time_steps, 2, 10)
    n_neurons = np.product(input_dimensions[2:])
    input_current = torch.ones(*input_dimensions) * 0.5 / (1-alpha)
    
    rec_weights = nn.Linear(n_neurons, n_neurons, bias=False)
    rec_weights.weight = nn.Parameter(torch.ones(n_neurons,n_neurons)/n_neurons*0.5/(1-alpha))
    layer = LIFRecurrent(alpha_mem=alpha, rec_weights=rec_weights)
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
