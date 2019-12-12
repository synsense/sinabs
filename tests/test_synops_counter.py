from sinabs import SynOpCounter
from sinabs.layers import NeuromorphicReLU
import torch
import numpy as np


class Model(torch.nn.Module):
    def __init__(self, quantize=False):
        super().__init__()

        self.seq = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=8,
                            kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=8, out_channels=12,
                            kernel_size=(3, 3), bias=False),
            NeuromorphicReLU(),
            torch.nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
            torch.nn.Conv2d(in_channels=12, out_channels=12,
                            kernel_size=(3, 3), bias=False),
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

        self.linear.weight.data = torch.tensor([[1.2, 1., 1., 3., -2.],
                                                [1.2, 1., 1., 2., -10.]])

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

    assert np.allclose(loss.item(), 4.)


def test_layer_synops():
    model = Model()
    criterion = SynOpCounter(model.modules(), sum_activations=False)

    input = torch.rand([1, 1, 64, 64])
    model(input)

    loss = criterion()

    assert len(loss) == 3
