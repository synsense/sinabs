import torch
from torch import nn
from sinabs.utils import normalize_weights


def test_normalize_weights():
    # init model
    cnn = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=(3, 3), bias=False),
        nn.BatchNorm2d(16, affine=True),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(16, 8, kernel_size=(3, 3), bias=True),
        nn.BatchNorm2d(8, affine=False, track_running_stats=True),
    )
    # get activation layers and parameter layers
    spike_lyrs = [name for name, child in cnn.named_children() if isinstance(child, (nn.ReLU, ))]
    param_lyrs = [name for name, child in cnn.named_children() if isinstance(child, (nn.Conv2d, nn.Linear))]
    # test on cpu tensor
    sample_input_cpu = torch.rand(8, 1, 28, 28)
    normalize_weights(cnn, sample_input_cpu, spike_lyrs, param_lyrs)
    # test on cuda tensor
    sample_input_cpu = torch.rand(8, 1, 28, 28).cuda()
    cnn = cnn.cuda()
    normalize_weights(cnn, sample_input_cpu, spike_lyrs, param_lyrs)

