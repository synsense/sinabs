def test_import():
    from sinabs.layers import SpikingConv3dLayer


def test_init():
    from sinabs.layers import SpikingConv3dLayer
    import torch

    # Generate input
    inp_spikes = (torch.rand((100, 2, 10, 15, 20)) > 0.95).float()

    # Init layer
    conv = SpikingConv3dLayer(
        channels_in=2,
        image_shape=(10, 15, 20),
        channels_out=7,
        kernel_shape=(2, 2, 2),
        padding=(0, 1, 0, 1, 0, 1),
    )

    out_spikes = conv(inp_spikes)

    assert out_spikes.shape == (100, 7, 10, 15, 20)
