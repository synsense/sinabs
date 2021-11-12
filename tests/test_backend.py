from copy import deepcopy


def test_backend_iaf():
    # Will only test trivial functionality. Test actual conversion in plugin unit tests.

    from sinabs.layers import IAF, IAFSqueeze

    layer = IAF()

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer_copy.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer_copy.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer_copy._param_dict[name] == val

    layer = IAFSqueeze(num_timesteps=10)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer_copy.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer_copy.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer_copy._param_dict[name] == val


def test_backend_lif():
    # Will only test trivial functionality. Test actual conversion in plugin unit tests.

    from sinabs.layers import LIF, LIFSqueeze

    layer = LIF(alpha_mem=0.8)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer_copy.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer_copy.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer_copy._param_dict[name] == val

    layer = LIFSqueeze(alpha_mem=0.8, num_timesteps=10)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    layer_copy = deepcopy(layer)

    # Backend remains the same. Therefore layer should not change.
    layer_sinabs_backend = layer.to_backend(layer.backend)
    assert layer is layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer_copy.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer_copy.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer_copy._param_dict[name] == val
