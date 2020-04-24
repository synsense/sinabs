def test_spikelayer_init():
    import torch
    from sinabs.layers.iaf_bptt import SpikingLayer

    layer = SpikingLayer(
        threshold=1,
        threshold_low=-1,
        membrane_subtract=True,
        batch_size=None,
    )

    inp = torch.rand((20,4,4))

    out = layer(inp)
    print(out.shape)