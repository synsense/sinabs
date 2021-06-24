import numpy as np


def test_spikelayer_init():
    import torch
    from sinabs.layers import IAF

    layer = IAF(threshold=1, threshold_low=-1)

    inp = torch.rand((20, 4, 4))

    out = layer(inp)
    print(out.shape)


def test_membrane_subtract():
    import torch
    from sinabs.layers import IAF

    layer = IAF(threshold=2.0, membrane_subtract=1.0)

    inp = torch.tensor([0.5, 0.6, 0.5, 0.5, 0.5]).reshape(5, 1)
    exp = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]).reshape(5, 1)  # one neuron, 5 time
    out = layer(inp)
    assert torch.equal(out, exp)
    assert layer.activations.item() == 0.0
    assert np.allclose(layer.state.item(), 1.6)


def test_membrane_subtract_multiple_spikes():
    import torch
    from sinabs.layers import IAF

    layer = IAF(threshold=2.0, membrane_subtract=2.5)

    inp = torch.tensor([[5.5], [0.0]])
    out = layer(inp)
    assert torch.equal(out, torch.tensor([[2.0], [0.0]]))
    assert layer.activations.item() == 0.0
    assert np.allclose(layer.state.item(), 0.5)


def test_membrane_reset():
    import torch
    from sinabs.layers import IAF

    layer = IAF(threshold=2.0, membrane_subtract=1.0, membrane_reset=True)

    inp = torch.tensor([0.5, 0.6, 0.5, 0.5, 0.5]).reshape(5, 1)
    exp = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]).reshape(5, 1)  # one neuron, 5 time
    out = layer(inp)
    assert torch.equal(out, exp)
    assert layer.activations.item() == 0.0
    assert np.allclose(layer.state.item(), 0.5)


def test_membrane_reset_multiple_spikes():
    import torch
    from sinabs.layers import IAF

    layer = IAF(threshold=2.0, membrane_reset=True)

    inp = torch.tensor([[5.5], [0.0]])
    out = layer(inp)
    assert torch.equal(out, torch.tensor([[1.0], [0.0]]))
    assert layer.activations.item() == 0.0
    assert np.allclose(layer.state.item(), 0.0)


def test_iaf_batching():
    import torch
    from sinabs.layers import IAFSqueeze, IAF

    batch_size = 7
    t_steps = 9
    n_neurons = 13

    # No squeeze
    sl = IAF(
        threshold=1.0, threshold_low=-1.0, membrane_subtract=1, membrane_reset=False
    )

    data = torch.rand((t_steps, batch_size, n_neurons))
    out_nosqueeze = sl(data)
    assert out_nosqueeze.shape == (t_steps, batch_size, n_neurons)

    # Squeeze, batch_size known
    sl = IAFSqueeze(
        threshold=1.0,
        threshold_low=-1.0,
        membrane_subtract=1,
        batch_size=batch_size,
        membrane_reset=False,
    )
    data_squeezed = data.movedim(0, 1).reshape(-1, n_neurons)
    out_batch = sl(data_squeezed)
    assert out_batch.shape == (t_steps * batch_size, n_neurons)

    # Squeeze, num_timesteps known
    sl = IAFSqueeze(
        threshold=1.0,
        threshold_low=-1.0,
        membrane_subtract=1,
        num_timesteps=t_steps,
        membrane_reset=False,
    )
    out_steps = sl(data_squeezed)
    assert out_steps.shape == (batch_size * t_steps, n_neurons)

    # Make sure all outputs are the same
    assert (out_steps == out_batch).all()
    out_unsqueezed = out_steps.reshape(batch_size, t_steps, n_neurons).movedim(0, 1)
    assert (out_unsqueezed == out_nosqueeze).all()
