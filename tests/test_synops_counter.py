import numpy as np
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
    input_ = torch.zeros((1, 3))
    input_[0, 0] = 3
    analyser = SNNAnalyzer(model)
    model(input_)
    model_stats = analyser.get_model_statistics()
    layer_stats = analyser.get_layer_statistics()[""]

    # 3 spikes * 5 channels
    assert model_stats["synops"] == 15
    assert layer_stats["synops"] == 15
    assert layer_stats["fanout_prev"] == 5


def test_linear_synops_counter_across_batches():
    model = nn.Linear(3, 5)
    input1 = torch.zeros((1, 3))
    input1[0, 0] = 3
    input2 = torch.zeros((1, 3))
    input2[0, 0] = 6
    analyser = SNNAnalyzer(model)
    model(input1)
    model(input2)
    model_stats = analyser.get_model_statistics()
    layer_stats = analyser.get_layer_statistics()[""]

    # (3+6)/2 spikes * 5 channels
    assert model_stats["synops"] == 22.5
    assert layer_stats["synops"] == 22.5
    assert layer_stats["fanout_prev"] == 5


def test_conv_synops_counter():
    model = nn.Conv2d(1, 5, kernel_size=2)
    input_ = torch.zeros((1, 1, 3, 3))
    input_[0, 0, 1, 1] = 3
    analyser = SNNAnalyzer(model)
    model(input_)
    model_stats = analyser.get_model_statistics()
    layer_stats = analyser.get_layer_statistics()[""]

    # 3 spikes, 2x2 kernel, 5 channels
    assert model_stats["synops"] == 60
    assert layer_stats["synops"] == 60
    assert layer_stats["fanout_prev"] == 20


def test_conv_synops_counter_counts_across_batches():
    model = nn.Conv2d(1, 5, kernel_size=2)
    input1 = torch.zeros((1, 1, 3, 3))
    input1[0, 0, 1, 1] = 6
    input2 = torch.zeros((1, 1, 3, 3))
    input2[0, 0, 1, 1] = 3
    analyser = SNNAnalyzer(model)
    model(input1)
    model(input2)
    model_stats = analyser.get_model_statistics()
    layer_stats = analyser.get_layer_statistics()[""]

    # (3+6)/2 spikes, 2x2 kernel, 5 channels
    assert model_stats["synops"] == 90
    assert layer_stats["synops"] == 90
    assert layer_stats["fanout_prev"] == 20


def test_spiking_layer_firing_rate():
    layer = sl.IAF()
    input_ = torch.eye(4).unsqueeze(0).unsqueeze(0)

    analyser = sinabs.SNNAnalyzer(layer)
    output = layer(input_)
    model_stats = analyser.get_model_statistics()
    layer_stats = analyser.get_layer_statistics()[""]

    assert (output == input_).all()
    assert model_stats["firing_rate"] == 0.25
    assert layer_stats["firing_rate"] == 0.25
    assert layer_stats["firing_rate_per_neuron"].shape == (4, 4)
    assert layer_stats["firing_rate_per_neuron"].mean() == 0.25


def test_spiking_layer_firing_rate_across_batches():
    layer = sl.IAF()
    input1 = torch.eye(4).unsqueeze(0).unsqueeze(0)
    input2 = 2 * torch.eye(4).unsqueeze(0).unsqueeze(0)

    analyser = sinabs.SNNAnalyzer(layer)
    output = layer(input1)
    sinabs.reset_states(layer)
    output = layer(input2)
    model_stats = analyser.get_model_statistics()
    layer_stats = analyser.get_layer_statistics()[""]

    assert (output == input2).all()
    assert model_stats["firing_rate"] == 0.375
    assert layer_stats["firing_rate"] == 0.375
    assert layer_stats["firing_rate_per_neuron"].shape == (4, 4)
    assert layer_stats["firing_rate_per_neuron"].mean() == 0.375


def test_analyser_reset():
    layer = sl.IAF()
    input_ = 2 * torch.eye(4).unsqueeze(0).unsqueeze(0)

    analyser = sinabs.SNNAnalyzer(layer)
    output = layer(input_)
    output = layer(input_)
    sinabs.reset_states(layer)
    analyser.reset()
    output = layer(input_)
    model_stats = analyser.get_model_statistics()
    layer_stats = analyser.get_layer_statistics()[""]

    assert (output == input_).all()
    assert model_stats["firing_rate"] == 0.5
    assert layer_stats["firing_rate"] == 0.5
    assert layer_stats["firing_rate_per_neuron"].shape == (4, 4)
    assert layer_stats["firing_rate_per_neuron"].mean() == 0.5


def test_snn_analyser_statistics():
    batch_size = 3
    num_timesteps = 10
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=2, bias=False),
        IAFSqueeze(batch_size=batch_size),
        nn.Conv2d(2, 3, kernel_size=2, bias=False),
        IAFSqueeze(batch_size=batch_size),
    )

    analyser = SNNAnalyzer(model)
    input_ = torch.rand((batch_size, num_timesteps, 1, 4, 4)) * 100
    input_flattended = input_.flatten(0, 1)
    output = model(input_flattended)
    layer_stats = analyser.get_layer_statistics()
    model_stats = analyser.get_model_statistics()

    # spiking layer checks
    assert (
        layer_stats["3"]["firing_rate"] == output.mean()
    ), "The output mean should be equivalent to the firing rate of the last spiking layer"
    assert (
        torch.cat(
            (
                layer_stats["1"]["firing_rate_per_neuron"].ravel(),
                layer_stats["3"]["firing_rate_per_neuron"].ravel(),
            )
        ).mean()
        == model_stats["firing_rate"]
    ), "Mean of layer 1 and 3 firing rates is not equal to calculated model firing rate."

    # parameter layer checks
    layer_stats["0"]["synops"] == input_.mean(0).sum() * np.product(
        model[0].kernel_size
    ) * model[0].out_channels
    assert layer_stats["0"]["num_timesteps"] == num_timesteps
    assert layer_stats["2"]["num_timesteps"] == num_timesteps
    assert (
        model_stats["synops"] == layer_stats["0"]["synops"] + layer_stats["2"]["synops"]
    )


def test_snn_analyser_does_not_depend_on_batch_size():
    batch_size_1 = 5
    num_timesteps = 10
    linear1 = nn.Linear(3, 4, bias=False)
    analyser = SNNAnalyzer(linear1)
    input_ = torch.ones((batch_size_1, num_timesteps, 3)) * 10
    linear1(input_)
    model_stats_batch_size_1 = analyser.get_model_statistics()

    batch_size_2 = 10
    linear2 = nn.Linear(3, 4, bias=False)
    analyser = SNNAnalyzer(linear2)
    input_ = torch.ones((batch_size_2, num_timesteps, 3)) * 10
    linear2(input_)
    model_stats_batch_size_2 = analyser.get_model_statistics()

    assert model_stats_batch_size_1["synops"] == model_stats_batch_size_2["synops"]
