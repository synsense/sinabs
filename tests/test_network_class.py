from sinabs.from_torch import from_model
from torch import nn
import torch
import numpy as np


ann = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=(3, 3), bias=False),
    nn.ReLU(),
)

network = from_model(ann)
data = torch.rand((1, 1, 4, 4))


def test_reset_states():
    network(data)
    assert network.spiking_model[1].state.sum() != 0
    network.reset_states()
    assert network.spiking_model[1].state.sum() == 0


def test_compare_activations():
    analog, rates = network.compare_activations(data)
    assert len(analog) == len(rates) == 3
    for anl, spk in zip(analog, rates):
        assert np.squeeze(anl).shape == np.squeeze(spk).shape
