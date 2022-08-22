import torch
import sinabs.layers as sil
import numpy as np
from torch import nn
from sinabs.from_torch import from_model
import pytest

from sinabs.layers import IAFSqueeze


def test_reconstruct_image():
    # generate random image
    img_shape = (3, 100, 100)
    image = 255.0 * np.random.random(size=img_shape)

    # instantiate layer
    spklayer = sil.Img2SpikeLayer(
        image_shape=img_shape, tw=10000, max_rate=1000.0, squeeze=True
    )

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
        image_shape=input_shape,
        tw=10000,
        max_rate=1000.0,
        squeeze=True,
        negative_spikes=True,
        norm=1.0,
    )

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

    cnn.sequence[1].running_mean = 2.0 * torch.rand(16) - 1.0
    cnn.sequence[1].running_var = torch.rand(16) + 1.0
    cnn.sequence[1].weight = nn.Parameter(2 * torch.rand(16) - 1.0)
    cnn.sequence[1].bias = nn.Parameter(2 * torch.rand(16) - 1.0)

    cnn.sequence[5].running_mean = 2.0 * torch.rand(8) - 1.0
    cnn.sequence[5].running_var = torch.rand(8) + 1.0

    img2spk = sil.Img2SpikeLayer(image_shape=input_shape, tw=1000, norm=1.0)
    snn = from_model(cnn, input_shape=input_shape)

    img = torch.Tensor(np.random.random(size=input_shape))

    with torch.no_grad():
        spk_img = img2spk(img)
        snn_res = snn(spk_img).mean(0)
        cnn_res = cnn(img.unsqueeze(0))

    # import matplotlib.pyplot as plt
    # plt.plot(snn_res.numpy().ravel(), cnn_res.numpy().ravel(), '.')
    # plt.show()

    assert np.allclose(snn_res, cnn_res, atol=0.025)


def test_network_conversion_add_spk_out():
    cnn = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=(3, 3), bias=False),
        nn.BatchNorm2d(16, affine=True),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(16, 8, kernel_size=(3, 3), bias=True),
        nn.BatchNorm2d(8, affine=False, track_running_stats=True),
    )

    input_shape = (1, 28, 28)

    cnn[1].running_mean = 2.0 * torch.rand(16) - 1.0
    cnn[1].running_var = torch.rand(16) + 1.0
    cnn[1].weight = nn.Parameter(2 * torch.rand(16) - 1.0)
    cnn[1].bias = nn.Parameter(2 * torch.rand(16) - 1.0)

    cnn[5].running_mean = 2.0 * torch.rand(8) - 1.0
    cnn[5].running_var = torch.rand(8) + 1.0

    img2spk = sil.Img2SpikeLayer(image_shape=input_shape, tw=1000, norm=1.0)

    snn = from_model(cnn, input_shape=input_shape, add_spiking_output=True)

    img = torch.Tensor(np.random.random(size=input_shape))

    with torch.no_grad():
        spk_img = img2spk(img)
        snn(spk_img).mean(0)


def test_network_conversion_complicated_model():
    """
    Try converting rather complicated network model with nested structures, which used
    to fail before.
    """

    ann = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.Sequential(nn.AvgPool2d(2), nn.ReLU(), nn.Conv2d(1, 1, 1)),
        nn.ReLU(),
        nn.AvgPool2d(3),
        nn.Sequential(
            nn.Conv2d(1, 1, 2), nn.Sequential(nn.Flatten(), nn.Linear(400, 12))
        ),
    )

    snn = from_model(ann)


def test_network_conversion_with_batch_size():
    ann = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.Sequential(nn.AvgPool2d(2), nn.ReLU(), nn.Conv2d(1, 1, 1)),
        nn.ReLU(),
        nn.AvgPool2d(3),
        nn.Sequential(
            nn.Conv2d(1, 1, 2), nn.Sequential(nn.Flatten(), nn.Linear(400, 12))
        ),
    )

    # Model conversion without input size specification
    snn = from_model(ann, batch_size=32)
    assert snn.spiking_model[2].batch_size == 32

    # Model conversion with input size specification
    snn = from_model(ann, input_shape=(1, 128, 128), batch_size=32)
    assert snn.spiking_model[2].batch_size == 32


def test_network_conversion_with_num_timesteps():
    ann = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.Sequential(nn.AvgPool2d(2), nn.ReLU(), nn.Conv2d(1, 1, 1)),
        nn.ReLU(),
        nn.AvgPool2d(3),
        nn.Sequential(
            nn.Conv2d(1, 1, 2), nn.Sequential(nn.Flatten(), nn.Linear(400, 12))
        ),
    )

    # Model conversion without input size specification
    snn = from_model(ann, num_timesteps=32)
    assert snn.spiking_model[2].num_timesteps == 32

    # Model conversion with input size specification
    snn = from_model(ann, input_shape=(1, 128, 128), num_timesteps=32)
    assert snn.spiking_model[2].num_timesteps == 32


def test_network_conversion_backend():
    """
    Try conversion with sinabs explicitly stated as backend.
    """

    ann = nn.Sequential(
        nn.Conv2d(1, 1, 1),
        nn.Sequential(nn.AvgPool2d(2), nn.ReLU(), nn.Conv2d(1, 1, 1)),
        nn.ReLU(),
        nn.AvgPool2d(3),
        nn.Sequential(
            nn.Conv2d(1, 1, 2), nn.Sequential(nn.Flatten(), nn.Linear(400, 12))
        ),
    )

    snn = from_model(ann, backend="sinabs")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_parameter_devices():
    cnn = nn.Sequential(nn.Conv2d(1, 16, kernel_size=(3, 3), bias=False), nn.ReLU())

    snn = from_model(cnn.cuda(), input_shape=None)

    spk_img = torch.rand((10, 1, 28, 28)).cuda()

    with torch.no_grad():
        snn(spk_img)


def test_activation_change():
    from sinabs.layers import SumPool2d

    ann = nn.Sequential(
        nn.Conv2d(1, 4, (3, 3), padding=1),
        nn.ReLU(),
        SumPool2d(2, 2),
        nn.Conv2d(1, 4, (3, 3), padding=1),
        nn.ReLU(),
    )
    from sinabs.activation import SingleSpike, MembraneReset, Heaviside

    network = from_model(
        ann,
        spike_threshold=3.0,
        spike_fn=SingleSpike(),
        reset_fn=MembraneReset(),
        min_v_mem=-3.0,
    )

    spk_layer: IAFSqueeze = network.spiking_model[1]

    # Test number of spikes generated against neuron threshold
    input_data = torch.zeros(10, 4, 32, 32)
    input_data[:, 0, 0, 0] = 1.0
    out = spk_layer(input_data)
    assert out.sum() == 10 // 3

    # Test  input much larger than threshold
    spk_layer.reset_states()
    input_data = torch.zeros(10, 4, 32, 32)
    input_data[0, 0, 0, 0] = 7
    out = spk_layer(input_data)
    assert out.sum() == 1

    # Test same for the second layer
    spk_layer: IAFSqueeze = network.spiking_model[4]

    # Test number of spikes generated against neuron threshold
    input_data = torch.zeros(10, 4, 32, 32)
    input_data[:, 0, 0, 0] = 1.0
    out = spk_layer(input_data)
    assert out.sum() == 10 // 3

    # Test  input much larger than threshold
    spk_layer.reset_states()
    input_data = torch.zeros(10, 4, 32, 32)
    input_data[0, 0, 0, 0] = 7
    out = spk_layer(input_data)
    assert out.sum() == 1
