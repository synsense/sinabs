from itertools import product
from pathlib import Path

import pytest
import torch
from torch import nn

import sinabs.layers as sl
from sinabs import hooks
from sinabs.layers import IAFSqueeze

TEST_DIR = Path(__file__).resolve().parent
INPUT_RESULT_DIR = TEST_DIR / "inputs_and_results" / "hooks"
MODEL_DIR = TEST_DIR / "models"


def test_linear_layer_synops_hook():
    model = nn.Linear(3, 5)
    inp = torch.zeros((2, 3))
    inp[0, 0] = 3
    inp[1, 2] = 5
    model.register_forward_hook(hooks.linear_layer_synops_hook)
    model(inp)

    # (3 + 5) spikes * 5 channels, over 2 samples
    assert model.hook_data["layer_synops_per_timestep"] == 20


kernel_sizes = (2, 3, 5)
strides = (1, 2)
paddings = (0, 1, 2)
combinations = product(kernel_sizes, strides, paddings)


@pytest.mark.parametrize("kernel_size,stride,padding", combinations)
def test_conv_layer_synops_hook(kernel_size, stride, padding):
    inp = torch.load(INPUT_RESULT_DIR / "conv_input.pth")
    correct_synops = torch.load(INPUT_RESULT_DIR / "conv_layer_synops.pth")
    model = nn.Conv2d(3, 5, kernel_size=kernel_size, stride=stride, padding=padding)
    model.register_forward_hook(hooks.conv_layer_synops_hook)
    model(inp)

    correct = correct_synops[(kernel_size, stride, padding)]
    assert model.hook_data["layer_synops_per_timestep"] == correct


dts = (None, 1, 0.1, 2)


@pytest.mark.parametrize("dt", dts)
def test_model_synops_hook(dt):
    inp = torch.load(INPUT_RESULT_DIR / "conv_input.pth")
    correct_synops = torch.load(INPUT_RESULT_DIR / "model_synops.pth")
    model = torch.load(MODEL_DIR / "synop_hook_model.pth")
    hooks.register_synops_hooks(model, dt=dt)

    model(inp)
    for idx, synops in correct_synops.items():
        assert model[idx].hook_data["synops_per_timestep"] == synops
        assert model.hook_data["synops_per_timestep"][idx] == synops
        if dt is not None:
            assert model[idx].hook_data["synops_per_second"] == synops / dt
            assert model.hook_data["synops_per_second"][idx] == synops / dt
    synops_total = sum(correct_synops.values())
    assert model.hook_data["total_synops_per_timestep"] == synops_total
    if dt is not None:
        assert model.hook_data["total_synops_per_second"] == synops_total / dt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("dt", dts)
def test_model_synops_hook_cuda(dt):
    inp = torch.load(INPUT_RESULT_DIR / "conv_input.pth").cuda()
    correct_synops = torch.load(
        INPUT_RESULT_DIR / "model_synops.pth", map_location="cuda"
    )
    model = torch.load(MODEL_DIR / "synop_hook_model.pth").cuda()
    hooks.register_synops_hooks(model, dt=dt)

    model(inp)
    for idx, synops in correct_synops.items():
        assert model[idx].hook_data["synops_per_timestep"] == synops
        if dt is not None:
            assert model[idx].hook_data["synops_per_second"] == synops / dt
    synops_total = sum(correct_synops.values())
    assert model.hook_data["total_synops_per_timestep"] == synops_total
    if dt is not None:
        assert model.hook_data["total_synops_per_second"] == synops_total / dt


def test_firing_rate_hook():
    inp = torch.load(INPUT_RESULT_DIR / "conv_input.pth")
    model = torch.load(MODEL_DIR / "synop_hook_model.pth")
    correct_firing_rates = torch.load(INPUT_RESULT_DIR / "firing_rates.pth")
    for layer in model:
        if isinstance(layer, IAFSqueeze):
            layer.register_forward_hook(hooks.firing_rate_hook)
    model(inp)
    for idx, firing_rate in correct_firing_rates.items():
        assert model[idx].hook_data["firing_rate"] == firing_rate


def test_firing_rate_per_neuron_hook():
    inp = torch.load(INPUT_RESULT_DIR / "conv_input.pth")
    model = torch.load(MODEL_DIR / "synop_hook_model.pth")
    correct_firing_rates = torch.load(INPUT_RESULT_DIR / "firing_rates_per_neuron.pth")
    for layer in model:
        if isinstance(layer, IAFSqueeze):
            layer.register_forward_hook(hooks.firing_rate_per_neuron_hook)
    model(inp)
    for idx, firing_rate in correct_firing_rates.items():
        assert (model[idx].hook_data["firing_rate_per_neuron"] == firing_rate).all()


def test_input_diff_hook():
    inp = torch.load(INPUT_RESULT_DIR / "conv_input.pth")
    model = torch.load(MODEL_DIR / "synop_hook_model.pth")
    correct_values = torch.load(INPUT_RESULT_DIR / "input_diffs.pth")
    for layer in model:
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            layer.register_forward_hook(hooks.input_diff_hook)
    model(inp)
    for idx, correct in correct_values.items():
        assert (model[idx].hook_data["diff_output"] == correct).all()
