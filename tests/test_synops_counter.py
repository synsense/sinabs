import math

import numpy as np
import pytest
import torch
from torch import nn

import sinabs
import sinabs.layers as sl
from sinabs import SNNAnalyzer, SynOpCounter
from sinabs.layers import IAFSqueeze, NeuromorphicReLU


class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__(
            torch.nn.Conv2d(
                in_channels=1, out_channels=8, kernel_size=(3, 3), bias=False
            ),
            NeuromorphicReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=8, out_channels=12, kernel_size=(3, 3), bias=False
            ),
            NeuromorphicReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(
                in_channels=12, out_channels=12, kernel_size=(3, 3), bias=False
            ),
            NeuromorphicReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Dropout(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(432, 10, bias=False),
            NeuromorphicReLU(fanout=0),
        )


class TinyModel(torch.nn.Module):
    def __init__(self, quantize):
        super().__init__()
        self.linear = torch.nn.Linear(5, 2, bias=False)
        self.relu = NeuromorphicReLU(fanout=2, quantize=quantize)

        self.linear.weight.data = torch.tensor(
            [[1.2, 1.0, 1.0, 3.0, -2.0], [1.2, 1.0, 1.0, 2.0, -10.0]]
        )

    def forward(self, x):
        return self.relu(self.linear(x))


def test_parsing():
    model = Model()
    loss = SynOpCounter(model.modules())

    assert len(loss.modules) == 3


def test_loss():
    model = TinyModel(quantize=False)
    criterion = SynOpCounter(model.modules())
    input = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    model(input)
    loss = criterion()

    assert np.allclose(loss.item(), 4.2)


def test_loss_quantized():
    model = TinyModel(quantize=True)
    criterion = SynOpCounter(model.modules())
    input = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5]])
    model(input)
    loss = criterion()

    assert np.allclose(loss.item(), 4.0)


def test_layer_synops():
    model = Model()
    criterion = SynOpCounter(model.modules(), sum_activations=False)
    input = torch.rand([1, 1, 64, 64])
    model(input)
    loss = criterion()

    assert len(loss) == 3


def test_linear_synops_counter():
    model = nn.Linear(3, 5)
    input_ = torch.zeros((2, 3))
    input_[0, 0] = 3
    input_[1, 2] = 5
    analyzer = SNNAnalyzer(model)
    model(input_)
    model_stats = analyzer.get_model_statistics()
    layer_stats = analyzer.get_layer_statistics()["parameter"][""]

    # (3 + 5) spikes * 5 channels, over 2 samples
    assert model_stats["synops"] == 20
    assert layer_stats["synops"] == 20
    assert layer_stats["synops/s"] == math.inf


def test_linear_synops_counter_with_time():
    model = nn.Linear(3, 5)
    n_steps, dt = 10, 10.0
    input_ = torch.zeros((2, n_steps, 3))
    input_[0, 0, 0] = 3
    input_[1, 0, 1] = 5
    analyzer = SNNAnalyzer(model, dt=dt)
    model(input_)
    layer_stats = analyzer.get_layer_statistics()["parameter"][""]

    assert layer_stats["synops"] == layer_stats["synops/s"] * n_steps * dt / 1000


def test_linear_synops_counter_across_batches():
    model = nn.Linear(3, 5)
    input1 = torch.zeros((1, 3))
    input1[0, 0] = 3
    input2 = torch.zeros((1, 3))
    input2[0, 0] = 6
    analyzer = SNNAnalyzer(model)
    model(input1)
    batch1_stats = analyzer.get_model_statistics(average=False)
    model(input2)
    batch2_stats = analyzer.get_model_statistics(average=False)
    model_stats = analyzer.get_model_statistics(average=True)
    layer_stats = analyzer.get_layer_statistics(average=True)["parameter"][""]

    # (3+6)/2 spikes * 5 channels
    assert model_stats["synops"] == 22.5
    assert layer_stats["synops"] == 22.5
    assert 2 * batch1_stats["synops"] == batch2_stats["synops"]


def test_conv_synops_counter():
    model = nn.Conv2d(1, 5, kernel_size=2)
    input_ = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
    analyzer = SNNAnalyzer(model)
    model(input_)
    model_stats = analyzer.get_model_statistics(average=True)
    layer_stats = analyzer.get_layer_statistics(average=True)["parameter"][""]

    # 1 spike x4, 2 spikes x1, all x5 output channels. Number is averaged across batch size
    assert model_stats["synops"] == 30
    assert layer_stats["synops"] == 30


def test_conv_synops_counter_counts_across_batches():
    model = nn.Conv2d(1, 5, kernel_size=2)
    input1 = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1)
    input2 = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1, 1) * 2
    analyzer = SNNAnalyzer(model)
    model(input1)
    batch1_stats = analyzer.get_model_statistics(average=False)
    model(input2)
    batch2_stats = analyzer.get_model_statistics(average=False)
    model_stats = analyzer.get_model_statistics(average=True)
    layer_stats = analyzer.get_layer_statistics(average=True)["parameter"][""]

    assert 2 * batch1_stats["synops"] == batch2_stats["synops"]
    assert model_stats["synops"] == 45
    assert layer_stats["synops"] == 45


def test_spiking_layer_firing_rate():
    layer = sl.IAF()
    input_ = torch.eye(4).unsqueeze(0).unsqueeze(0)

    analyzer = sinabs.SNNAnalyzer(layer)
    output = layer(input_)
    model_stats = analyzer.get_model_statistics(average=True)
    layer_stats = analyzer.get_layer_statistics(average=True)["spiking"][""]

    assert (output == input_).all()
    assert model_stats["firing_rate"] == 0.25
    assert layer_stats["firing_rate"] == 0.25
    assert layer_stats["firing_rate_per_neuron"].shape == (4, 4)
    assert layer_stats["firing_rate_per_neuron"].mean() == 0.25


def test_spiking_layer_firing_rate_across_batches():
    layer = sl.IAF()
    input1 = torch.eye(4).unsqueeze(0).unsqueeze(0)
    input2 = 2 * torch.eye(4).unsqueeze(0).unsqueeze(0)

    analyzer = sinabs.SNNAnalyzer(layer)
    output = layer(input1)
    batch1_stats = analyzer.get_model_statistics(average=False)
    sinabs.reset_states(layer)
    output = layer(input2)
    batch2_stats = analyzer.get_model_statistics(average=False)
    model_stats = analyzer.get_model_statistics(average=True)
    layer_stats = analyzer.get_layer_statistics(average=True)["spiking"][""]

    assert (output == input2).all()
    assert 2 * batch1_stats["firing_rate"] == batch2_stats["firing_rate"]
    assert model_stats["firing_rate"] == 0.375
    assert layer_stats["firing_rate"] == 0.375
    assert layer_stats["firing_rate_per_neuron"].shape == (4, 4)
    assert layer_stats["firing_rate_per_neuron"].mean() == 0.375


def test_analyzer_reset():
    layer = sl.IAF()
    input_ = 2 * torch.eye(4).unsqueeze(0).unsqueeze(0)

    analyzer = sinabs.SNNAnalyzer(layer)
    output = layer(input_)
    output = layer(input_)
    sinabs.reset_states(layer)
    analyzer.reset()
    output = layer(input_)
    model_stats = analyzer.get_model_statistics(average=True)
    layer_stats = analyzer.get_layer_statistics(average=True)["spiking"][""]

    assert (output == input_).all()
    assert model_stats["firing_rate"] == 0.5
    assert layer_stats["firing_rate"] == 0.5
    assert layer_stats["firing_rate_per_neuron"].shape == (4, 4)
    assert layer_stats["firing_rate_per_neuron"].mean() == 0.5


def test_snn_analyzer_statistics():
    batch_size = 3
    num_timesteps = 10
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=2, padding=3, stride=2, bias=False),
        IAFSqueeze(batch_size=batch_size),
        nn.Conv2d(2, 3, kernel_size=2, bias=False),
        IAFSqueeze(batch_size=batch_size),
    )

    analyzer = SNNAnalyzer(model)
    input_ = torch.rand((batch_size, num_timesteps, 1, 16, 16)) * 100
    input_flattended = input_.flatten(0, 1)
    output = model(input_flattended)
    spike_layer_stats = analyzer.get_layer_statistics(average=True)["spiking"]
    param_layer_stats = analyzer.get_layer_statistics(average=True)["parameter"]
    model_stats = analyzer.get_model_statistics(average=True)

    # spiking layer checks
    assert spike_layer_stats["3"]["input"].shape[0] == batch_size
    assert spike_layer_stats["3"]["input"].shape[1] == num_timesteps
    assert (
        spike_layer_stats["3"]["firing_rate"] == output.mean()
    ), "The output mean should be equivalent to the firing rate of the last spiking layer"
    assert (
        torch.cat(
            (
                spike_layer_stats["1"]["firing_rate_per_neuron"].ravel(),
                spike_layer_stats["3"]["firing_rate_per_neuron"].ravel(),
            )
        ).mean()
        == model_stats["firing_rate"]
    ), "Mean of layer 1 and 3 firing rates is not equal to calculated model firing rate."

    # parameter layer checks
    param_layer_stats["0"]["synops"] == input_.mean(0).sum() * np.product(
        model[0].kernel_size
    ) * model[0].out_channels
    assert param_layer_stats["0"]["num_timesteps"] == num_timesteps
    assert param_layer_stats["2"]["num_timesteps"] == num_timesteps
    assert (
        model_stats["synops"]
        == param_layer_stats["0"]["synops"] + param_layer_stats["2"]["synops"]
    )


def test_snn_analyzer_does_not_depend_on_batch_size():
    batch_size_1 = 5
    num_timesteps = 10
    linear1 = nn.Linear(3, 4, bias=False)
    analyzer = SNNAnalyzer(linear1)
    input_ = torch.ones((batch_size_1, num_timesteps, 3)) * 10
    linear1(input_)
    model_stats_batch_size_1 = analyzer.get_model_statistics(average=True)

    batch_size_2 = 10
    linear2 = nn.Linear(3, 4, bias=False)
    analyzer = SNNAnalyzer(linear2)
    input_ = torch.ones((batch_size_2, num_timesteps, 3)) * 10
    linear2(input_)
    model_stats_batch_size_2 = analyzer.get_model_statistics(average=True)

    assert model_stats_batch_size_1["synops"] == model_stats_batch_size_2["synops"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_snnanalyzer_on_gpu():
    linear = nn.Linear(3, 4, bias=False)
    analyzer = SNNAnalyzer(linear)
    linear.cuda()
    input_ = torch.ones((2, 10, 3), device="cuda") * 10
    linear(input_)
    model_stats_batch_size_1 = analyzer.get_model_statistics()
