import torch
import sinabs.layers as sil
import numpy as np
from torch import nn
from sinabs.from_torch import SpkConverter


def test_reconstruct_image():
    # generate random image
    img_shape = (3, 100, 100)
    image = 255.0 * np.random.random(size=img_shape)

    # instantiate layer
    spklayer = sil.Img2SpikeLayer(
        image_shape=img_shape, tw=10000, max_rate=1000.,
        squeeze_batch_dimension=True)

    spikes = spklayer(torch.Tensor(image))
    rates = spikes.mean(0).unsqueeze(0)

    # accept errors of 0.025 for numbers in (0, 1), over 10000 tsteps
    assert np.allclose(rates, image / 255.0, atol=0.025)


def test_reconstruct_real_numbers():
    # generate random image
    input_shape = (3, 100, 100)
    input_data = 2 * np.random.random(size=input_shape) - 1

    # instantiate layer
    spklayer = sil.Img2SpikeLayer(
        image_shape=input_shape, tw=10000, max_rate=1000.,
        squeeze_batch_dimension=True, negative_spikes=True,
        norm=1.0)

    spikes = spklayer(torch.Tensor(input_data))
    rates = spikes.mean(0).unsqueeze(0)

    # accept errors of 0.025 for numbers in (0, 1), over 10000 tsteps
    assert np.allclose(rates, input_data, atol=0.025)


def test_network_conversion():
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.sequence = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(3, 3), bias=False),
                nn.BatchNorm2d(16, affine=True),
                nn.ReLU(),
                nn.AvgPool2d(kernel_size=2),
                nn.Conv2d(16, 8, kernel_size=(3, 3), bias=True),
                nn.BatchNorm2d(8, affine=False, track_running_stats=True),
            )

        def forward(self, x):
            return self.sequence(x)

    input_shape = (1, 28, 28)

    cnn = CNN().eval()
    img2spk = sil.Img2SpikeLayer(image_shape=input_shape, tw=1000, norm=1.)
    converter = SpkConverter(cnn, input_shape=input_shape,
                             input_conversion_layer=img2spk)
    snn = converter.convert()

    img = torch.Tensor(np.random.random(size=input_shape))

    with torch.no_grad():
        snn_res = snn(img).mean(0)
        cnn_res = cnn(img.unsqueeze(0))

    # import matplotlib.pyplot as plt
    # plt.plot(snn_res.numpy().ravel(), cnn_res.numpy().ravel(), '.')
    # plt.show()

    assert np.allclose(snn_res, cnn_res, atol=0.025)
