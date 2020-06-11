"""
This should test some individual cases of networks with properties that are
supported (but maybe not always common). Running these tests should not require
samna, and they are tests of equivalence between snn and speck compatible net.
"""
from backend.Speck import SpeckCompatibleNetwork
import torch
from torch import nn
from sinabs.from_torch import from_model
import numpy as np

input_shape = (2, 16, 16)
input = torch.rand(1, *input_shape, requires_grad=False) * 100.


def networks_equal_output(input, ann):
    snn = from_model(ann)
    snn.eval()
    snn_out = snn(input)  # forward pass

    snn.reset_states()
    spn = SpeckCompatibleNetwork(
        snn, input_shape=input.shape[1:], discretize=False
    )
    spn_out = spn(input)

    return np.array_equal(snn_out, spn_out)


def test_initial_pooling():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.seq(x)

    assert networks_equal_output(input, Net())


def test_pooling_consolidation():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.AvgPool2d(kernel_size=3, stride=3),
                nn.Conv2d(4, 2, kernel_size=1, stride=1),
                nn.ReLU()
            )

        def forward(self, x):
            return self.seq(x)

    assert networks_equal_output(input, Net())


def test_batchnorm_after_conv():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.AvgPool2d(kernel_size=3, stride=3),
                nn.Conv2d(4, 2, kernel_size=1, stride=1),
                nn.BatchNorm2d(2),
                nn.ReLU()
            )

        def forward(self, x):
            return self.seq(x)

    net = Net()

    # setting batchnorm parameters, otherwise it's just identity
    net.seq[-2].running_mean.data = torch.tensor([1.2, -1.5])
    net.seq[-2].running_var.data = torch.tensor([1.1, 0.7])
    net.seq[-2].weight.data = torch.tensor([-1.2, -3.5])
    net.seq[-2].bias.data = torch.tensor([-0.2, 0.3])

    assert networks_equal_output(input, net)
