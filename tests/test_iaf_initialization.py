import numpy as np


def test_spikelayer_init():
    import torch
    from sinabs.layers import SpikingLayer

    layer = SpikingLayer(
        threshold=1,
        threshold_low=-1,
        batch_size=None,
    )

    inp = torch.rand((20, 4, 4))

    out = layer(inp)
    print(out.shape)


def test_membrane_subtract():
    import torch
    from sinabs.layers import SpikingLayer

    layer = SpikingLayer(
        threshold=2.,
        membrane_subtract=1.,
    )

    inp = torch.tensor([0.5, 0.6, 0.5, 0.5, 0.5]).reshape(5, 1)
    exp = torch.tensor([0., 0., 0., 1., 0.]).reshape(5, 1)  # one neuron, 5 time
    out = layer(inp)
    assert torch.equal(out, exp)
    assert layer.activations.item() == 0.
    assert np.allclose(layer.state.item(), 1.6)


def test_membrane_subtract_multiple_spikes():
    import torch
    from sinabs.layers import SpikingLayer

    layer = SpikingLayer(
        threshold=2.,
        membrane_subtract=2.5,
    )

    inp = torch.tensor([[5.5], [0.0]])
    out = layer(inp)
    assert torch.equal(out, torch.tensor([[2.0], [0.0]]))
    assert layer.activations.item() == 0.0
    assert np.allclose(layer.state.item(), 0.5)


def test_membrane_reset():
    import torch
    from sinabs.layers import SpikingLayer

    layer = SpikingLayer(
        threshold=2.,
        membrane_subtract=1.,
        membrane_reset=True,
    )

    inp = torch.tensor([0.5, 0.6, 0.5, 0.5, 0.5]).reshape(5, 1)
    exp = torch.tensor([0., 0., 0., 1., 0.]).reshape(5, 1)  # one neuron, 5 time
    out = layer(inp)
    assert torch.equal(out, exp)
    assert layer.activations.item() == 0.
    assert np.allclose(layer.state.item(), 0.5)


def test_membrane_reset_multiple_spikes():
    import torch
    from sinabs.layers import SpikingLayer

    layer = SpikingLayer(
        threshold=2.,
        membrane_reset=True,
    )

    inp = torch.tensor([[5.5], [0.0]])
    out = layer(inp)
    assert torch.equal(out, torch.tensor([[1.0], [0.0]]))
    assert layer.activations.item() == 0.0
    assert np.allclose(layer.state.item(), 0.0)
