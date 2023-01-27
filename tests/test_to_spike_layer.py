import numpy as np
import torch

import sinabs.layers as sl


def test_reconstruct_image():
    # generate random image
    img_shape = (3, 20, 20)
    image = 255.0 * np.random.random(size=img_shape)

    # instantiate layer
    spklayer = sl.Img2SpikeLayer(
        image_shape=img_shape, tw=10000, max_rate=1000.0, squeeze=True
    )

    spikes = spklayer(torch.Tensor(image))
    rates = spikes.mean(0).unsqueeze(0)

    # accept errors of 0.025 for numbers in (0, 1), over 10000 tsteps
    assert np.allclose(rates, image / 255.0, atol=0.025)


def test_reconstruct_real_numbers():
    # generate random image
    input_shape = (3, 20, 20)
    input_data = 2 * np.random.random(size=input_shape) - 1

    # instantiate layer
    spklayer = sl.Img2SpikeLayer(
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
