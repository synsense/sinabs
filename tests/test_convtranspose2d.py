def test_import():
    """
    Basic syntax check test
    """
    from sinabs.layers import SpikingConvTranspose2dLayer


def test_TorchSpikingConvTranspose2dLayer_initialization():
    """
    Test initialization of ConvTranspose2dlayer layer
    :return:
    """
    from sinabs.layers import SpikingConvTranspose2dLayer

    lyr = SpikingConvTranspose2dLayer(
        channels_in=2,
        image_shape=(10, 10),
        channels_out=6,
        kernel_shape=(3, 3),
        strides=(1, 1),
        padding=(1, 1, 0, 0),
        bias=False,
        threshold=8,
        threshold_low=-8,
        membrane_subtract=8,
        layer_name="convtranspose2d",
    )

    assert lyr.output_shape == (6, 12, 14)


def test_getoutput_shape():
    from sinabs.layers import SpikingConvTranspose2dLayer
    import torch

    lyr = SpikingConvTranspose2dLayer(
        channels_in=2,
        image_shape=(10, 20),
        channels_out=6,
        kernel_shape=(3, 5),
        strides=(1, 1),
        padding=(0, 3, 6, 0),
        bias=False,
        threshold=8,
        threshold_low=-8,
        membrane_subtract=8,
        layer_name="convtranspose2d",
    )

    tsrInput = (torch.rand(10, 2, 10, 20) > 0.9).float()

    tsrOutput = lyr(tsrInput)
    assert lyr.output_shape == tsrOutput.shape[1:]
