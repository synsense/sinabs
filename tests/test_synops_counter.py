from sinabs import SynOpCounter, SNNSynOpCounter
from sinabs.layers import NeuromorphicReLU, IAF
import torch
import numpy as np
from torch import nn


class Model(torch.nn.Module):
    def __init__(self, quantize=False):
        super().__init__()

        self.seq = torch.nn.Sequential(
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
            torch.nn.Dropout2d(0.5),
            torch.nn.Flatten(),
            torch.nn.Linear(432, 10, bias=False),
            NeuromorphicReLU(fanout=0),
        )

    def forward(self, x):
        return self.seq(x)


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


def test_snn_synops_counter():
    model = nn.Sequential(nn.Conv2d(1, 5, kernel_size=2), IAF())

    inp = torch.tensor([[[[0, 0, 0], [0, 3, 0], [0, 0, 0]]]]).float()

    counter = SNNSynOpCounter(model)
    model(inp)
    # 3 spikes, 2x2 kernel, 5 channels
    assert counter.get_synops()["SynOps"].sum() == 60
    assert counter.get_total_synops() == 60
