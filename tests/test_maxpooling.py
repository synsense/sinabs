def test_maxpool2d():
    import torch

    from sinabs.layers import SpikingMaxPooling2dLayer

    inp = torch.zeros((2, 2, 10, 10))
    maxpool = SpikingMaxPooling2dLayer(pool_size=(2, 2), strides=(2, 2))

    # Verify output shape
    assert maxpool(inp).shape == (2, 2, 5, 5)


def test_maxpool_function():
    import torch

    from sinabs.layers import SpikingMaxPooling2dLayer

    lyr = SpikingMaxPooling2dLayer(pool_size=(2, 3), strides=None)

    tsrInput = (torch.rand(10, 1, 2, 3) > 0.8).float()
    # print(tsrInput.sum(0))
    tsrInput[:, 0, 0, 2] = 1
    tsrOut = lyr(tsrInput)

    assert tsrOut.sum().item() == 10.0
