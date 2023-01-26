def test_larger_batch_size_next():
    import torch
    import sinabs.layers as sl

    layer = sl.IAF()

    first_batch = torch.zeros(size=(2, 10, 1, 28, 28)) 
    second_batch = torch.zeros(size=(4, 10, 1, 28, 28))

    first_out = layer(first_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([first_batch.shape[0], *first_batch.shape[2:]])
    assert first_out.shape[0] == first_batch.shape[0]
    second_out = layer(second_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([second_batch.shape[0], *second_batch.shape[2:]])
    assert second_out.shape[0] == second_batch.shape[0]

def test_smaller_batch_size_next():
    import torch
    import sinabs.layers as sl

    layer = sl.IAF()
    first_batch = torch.ones(size=(4, 10, 1, 28, 28)) 
    second_batch = torch.ones(size=(2, 10, 1, 28, 28))

    first_out = layer(first_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([first_batch.shape[0], *first_batch.shape[2:]])
    assert first_out.shape[0] == first_batch.shape[0]
    second_out = layer(second_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([second_batch.shape[0], *second_batch.shape[2:]])
    assert second_out.shape[0] == second_batch.shape[0]

def test_same_batch_size_next():
    import torch
    import sinabs.layers as sl

    layer = sl.IAF()

    first_batch = torch.zeros(size=(2, 10, 1, 28, 28)) 
    second_batch = torch.zeros(size=(2, 10, 1, 28, 28))

    first_out = layer(first_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([first_batch.shape[0], *first_batch.shape[2:]])
    assert first_out.shape[0] == first_batch.shape[0]
    second_out = layer(second_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([second_batch.shape[0], *second_batch.shape[2:]])
    assert second_out.shape[0] == second_batch.shape[0]

def test_trailing_dim_change():
    import torch
    import sinabs.layers as sl

    layer = sl.IAF()

    first_batch = torch.zeros(size=(2, 10, 1, 28, 28)) 
    second_batch = torch.zeros(size=(2, 10, 1, 26, 26))

    first_out = layer(first_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([first_batch.shape[0], *first_batch.shape[2:]])
    assert first_out.shape[0] == first_batch.shape[0]
    second_out = layer(second_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([second_batch.shape[0], *second_batch.shape[2:]])
    assert second_out.shape[0] == second_batch.shape[0]
    