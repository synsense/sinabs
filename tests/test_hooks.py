from itertools import product

import pytest
import torch
from torch import nn

import sinabs.layers as sl
from sinabs import hooks
from sinabs.layers import IAFSqueeze


def test_linear_layer_synops_hook():
    model = nn.Linear(3, 5)
    inp = torch.zeros((2, 3))
    inp[0, 0] = 3
    inp[1, 2] = 5
    model.register_forward_hook(hooks.linear_layer_synops_hook)
    model(inp)

    # (3 + 5) spikes * 5 channels, over 2 samples
    assert model.layer_synops_per_timestep == 20

kernel_sizes = (2, 3, 5)
strides = (1, 2)
paddings = (0, 1, 2)
combinations = product(kernel_sizes, strides, paddings)

@pytest.mark.parametrize("kernel_size,stride,padding", combinations)
def test_conv_layer_synops_hook(kernel_size, stride, padding):
    inp = torch.load("inputs_and_results/hooks/conv_input.pth")
    correct_synops = torch.load("inputs_and_results/hooks/conv_layer_synops.pth")
    model = nn.Conv2d(3, 5, kernel_size=kernel_size, stride=stride, padding=padding)
    model.register_forward_hook(hooks.conv_layer_synops_hook)
    model(inp)

    assert model.layer_synops_per_timestep == correct_synops[(kernel_size, stride, padding)]

@pytest.mark.parametrize("dt", (None, 1, 0.1, 2))
def test_model_synops_hook(dt):
    inp = torch.load("inputs_and_results/hooks/conv_input.pth")
    correct_synops = torch.load("inputs_and_results/hooks/model_synops.pth")
    model = torch.load("models/synop_hook_model.pth")
    hooks.register_synops_hooks(model, dt=dt)

    model(inp)
    for idx, synops in correct_synops.items():
        assert model[idx].synops_per_timestep == synops
        if dt is not None:
            assert model[idx].synops_per_second == synops / dt

