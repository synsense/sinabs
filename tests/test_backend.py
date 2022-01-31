import torch
from copy import deepcopy


def compare_layers(lyr0, lyr1):
    for p0, p1 in zip(lyr0.parameters(), lyr1.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(lyr0.buffers(), lyr1.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1
    for name, val in lyr0._param_dict.items():
        assert lyr1._param_dict[name] == val


def test_backend_iaf():
    # Will only test trivial functionality. Test actual conversion in plugin unit tests.

    from sinabs.layers import IAF, IAFSqueeze

    layer = IAF()

    layer(torch.rand(10, 10, 10))

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_copy, layer_sinabs_backend)

    layer = IAFSqueeze(num_timesteps=10)

    layer(torch.rand(10, 10, 10))

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_copy, layer_sinabs_backend)


def test_backend_lif():
    # Will only test trivial functionality. Test actual conversion in plugin unit tests.

    from sinabs.layers import LIF, LIFSqueeze

    layer = LIF(tau_mem=20.0)

    layer(torch.rand(10, 10, 10))

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_copy, layer_sinabs_backend)

    layer = LIFSqueeze(tau_mem=20.0, num_timesteps=10)

    layer(torch.rand(10, 10, 10))

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_copy, layer_sinabs_backend)
