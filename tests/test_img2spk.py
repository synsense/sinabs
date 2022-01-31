def test_img2spk():
    import torch
    from sinabs.layers import Img2SpikeLayer

    lyr = Img2SpikeLayer(
        image_shape=(2, 64, 64),
        tw=10,
        max_rate=1000,
    )

    img = torch.rand(2, 64, 64)

    spks = lyr(img)

    assert spks.shape == (10, 2, 64, 64)
