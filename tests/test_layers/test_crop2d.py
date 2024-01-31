def test_crop2d():
    import torch

    from sinabs.layers import Cropping2dLayer

    sinabs_layer = Cropping2dLayer(((2, 3), (3, 1)))
    inp = torch.zeros((2, 2, 10, 10))
    assert sinabs_layer(inp).shape == (2, 2, 5, 6)
