"""
This should test some individual cases of networks with properties that are
supported (but maybe not always common). Running these tests should not require
samna, and they are tests of equivalence between snn and speck compatible net.
"""
from sinabs.backend.speck import SpeckCompatibleNetwork
import torch
from torch import nn
from sinabs.from_torch import from_model
from sinabs.layers.iaf_bptt import SpikingLayer

input_shape = (2, 16, 16)
input_data = torch.rand(1, *input_shape, requires_grad=False) * 100.

TEST_CONFIGS = True  # set to False if testing without samna installed.


# --- UTILITIES --- #
def reset_states(seq):
    for s in seq:
        if isinstance(s, SpikingLayer):
            s.reset_states()
    return seq


def networks_equal_output(input_data, snn):
    snn.eval()
    snn_out = snn(input_data).squeeze()  # forward pass
    reset_states(snn)

    # snn.reset_states()
    spn = SpeckCompatibleNetwork(
        snn, input_shape=input_data.shape[1:], discretize=False
    )
    spn_out = spn(input_data).squeeze()

    assert torch.equal(snn_out, spn_out)

    if TEST_CONFIGS:
        # this will give an error if the config is not compatible
        spn.make_config()


# --- TESTS --- #
def test_with_class():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(
                nn.Conv2d(2, 4, kernel_size=2, stride=2),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.seq(x)

    snn = from_model(Net())
    snn.eval()
    snn_out = snn(input_data).squeeze()  # forward pass

    snn.reset_states()
    spn = SpeckCompatibleNetwork(
        snn, input_shape=input_data.shape[1:], discretize=False
    )
    spn_out = spn(input_data).squeeze()

    assert torch.equal(snn_out, spn_out)


def test_with_sinabs_batch():
    seq = nn.Sequential(
        nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
        nn.Conv2d(2, 4, kernel_size=2, stride=2),
        SpikingLayer(batch_size=1),
    )

    networks_equal_output(input_data, seq)


def test_initial_pooling():
    seq = nn.Sequential(
        nn.AvgPool2d(kernel_size=(2, 1), stride=(2, 1)),
        nn.Conv2d(2, 4, kernel_size=2, stride=2),
        SpikingLayer(),
    )

    networks_equal_output(input_data, seq)


def test_pooling_consolidation():
    seq = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=2, stride=2),
        SpikingLayer(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(4, 2, kernel_size=1, stride=1),
        SpikingLayer()
    )

    networks_equal_output(input_data, seq)


def test_different_xy_input():
    input_shape = (2, 16, 32)
    input_data = torch.rand(1, *input_shape, requires_grad=False) * 100.

    seq = nn.Sequential(
        nn.Conv2d(2, 4, kernel_size=2, stride=2),
        SpikingLayer(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(4, 2, kernel_size=1, stride=1),
        SpikingLayer()
    )

    networks_equal_output(input_data, seq)


def test_batchnorm_after_conv():
    seq = nn.Sequential(
        nn.Conv2d(2, 2, kernel_size=1, stride=1),
        nn.BatchNorm2d(2),
        SpikingLayer()
    )

    # setting batchnorm parameters, otherwise it's just identity
    seq[-2].running_mean.data = torch.tensor([1.2, -1.5])
    seq[-2].running_var.data = torch.tensor([1.1, 0.7])
    seq[-2].weight.data = torch.tensor([-1.2, -3.5])
    seq[-2].bias.data = torch.tensor([-0.2, 0.3])

    networks_equal_output(input_data, seq)


def test_flatten_linear():
    seq = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 2),
        SpikingLayer()
    )

    networks_equal_output(input_data, seq)
