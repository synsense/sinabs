import torch
from sinabs.layers import ExpLeak, ExpLeakSqueeze
import numpy as np
import pytest


def test_leaky():
    time_steps = 100
    input_current = torch.rand(time_steps, 2, 7, 7)
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
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7)
    input_current[:,0] = 1 # only inject current in the first time step
    layer = ExpLeak(tau_mem=torch.tensor(time_steps))
    membrane_output = layer(input_current)

    # first time step is not decayed
    membrane_decay = torch.exp(torch.tensor(-(time_steps-1)/time_steps))
    # account for rounding errors with .isclose()
    assert (membrane_output[:,0] == 1).all(), "Output for first time step is not correct."
    assert (membrane_output[:,-1] == layer.state).all(), "Output of last time step does not correspond to last layer state."
    assert torch.isclose(layer.state, membrane_decay, atol=1e-08).all(), "Neuron membrane potentials do not seems to decay correctly."

