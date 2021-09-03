import torch
from sinabs.layers import ExpLeak, ExpLeakSqueeze
import numpy as np
import pytest


def test_leaky():
    batch_size = 10
    time_steps = 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7)
    layer = ExpLeak(tau_mem=torch.tensor(1))
    membrane_output = layer(input_current)

    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0

def test_leaky_squeezed():
    batch_size = 10
    time_steps = 100
    input_current = torch.rand(batch_size*time_steps, 2, 7, 7)
    layer = ExpLeakSqueeze(tau_mem=torch.tensor(1), batch_size=batch_size)
    membrane_output = layer(input_current)

    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0

def test_leaky_membrane_decay():
    batch_size = 10
    time_steps = 100
    input_current = torch.ones(batch_size, time_steps, 2, 7, 7)
    input_current[:,1:,:,:] = 0 # only inject current in the first time step
    layer = ExpLeak(tau_mem=torch.tensor(time_steps))
    membrane_output = layer(input_current)

    # first time step is not decayed
    membrane_decay = torch.exp(torch.tensor(-(time_steps-1)/time_steps))
    # account for rounding errors with .isclose()
    assert torch.isclose(layer.state, membrane_decay, atol=1e-08).all(), "Neuron membrane potentials do not seems to decay correctly."

