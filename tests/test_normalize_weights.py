import torch
import pytest
from torch import nn
from sinabs.utils import normalize_weights


# init model
CNN = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=(3, 3), bias=False),
    nn.BatchNorm2d(16, affine=True),
    nn.ReLU(),
    nn.AvgPool2d(kernel_size=2),
    nn.Conv2d(16, 8, kernel_size=(3, 3), bias=True),
    nn.BatchNorm2d(8, affine=False, track_running_stats=True),
)

# get activation layers and parameter layers
SPIKE_LAYERS = [
    name for name, child in CNN.named_children() if isinstance(child, (nn.ReLU,))
]
PARAM_LAYERS = [
    name
    for name, child in CNN.named_children()
    if isinstance(child, (nn.Conv2d, nn.Linear))
]


def test_normalize_weights_cpu():
    # test on cpu tensor
    sample_input_cpu = torch.rand(8, 1, 28, 28)
    normalize_weights(
        CNN.cpu(), sample_input_cpu, SPIKE_LAYERS, SPIKE_LAYERS, percentile=100
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_normalize_weights_gpu():
    # test on cuda tensor
    sample_input_gpu = torch.rand(8, 1, 28, 28).cuda()
    normalize_weights(
        CNN.cuda(), sample_input_gpu, SPIKE_LAYERS, SPIKE_LAYERS, percentile=100
    )
