def test_forward():
    import torch
    from sinabs.layers import Sig2SpikeLayer

    channels = 4
    tw = 5

    lyr = Sig2SpikeLayer(
        channels_in=channels,
        tw=tw,
    )

    sig = torch.tensor([[1.0, 0.5, 0.25]] * channels)
    spk = lyr(sig)
    assert spk.shape == (tw * 3, channels)
