import pytest

@pytest.mark.parametrize(
    "first_batch_size,second_batch_size", [(2, 4), (4, 2), (2, 2)]
)
def test_batch_size_mismatch(first_batch_size, second_batch_size):
    import torch
    import sinabs.layers as sl
    from copy import copy

    # Setting the spike_threshold larger than the number of timestamps, such that
    # using `torch.ones()` will not cause any output spikes, thus resetting states
    n_timesteps = 10
    spike_threshold = 50
    tested_state = "v_mem"

    # StatefulLayer -> LIF -> IAF: Testing IAF tests all.
    layer = sl.IAF(spike_threshold=spike_threshold)

    first_batch = torch.ones(size=(first_batch_size, n_timesteps, 1, 28, 28)) 
    second_batch = torch.ones(size=(second_batch_size, n_timesteps, 1, 28, 28))

    first_out = layer(first_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([first_batch_size, *first_batch.shape[2:]])
    assert first_out.shape[0] == first_batch.shape[0]
   
    state_after_first_batch = None 
    for name, buff in layer.named_buffers():
        if name == tested_state:
            state_after_first_batch = copy(buff) 
            break 
    # once the second batch is passed the states should not be zero
    second_out = layer(second_batch)
    for buff in layer.buffers():
        assert buff.size() == torch.Size([second_batch_size, *second_batch.shape[2:]])
    assert second_out.shape[0] == second_batch.shape[0]

    state_after_second_batch = None
    for name, buff in layer.named_buffers():
        if name == tested_state:
            state_after_second_batch = copy(buff)
            break

    min_batch_size = min(first_batch_size, second_batch_size)
    
    # previously statement == 0, due to reset for case the cases first_batch_size != second_batch_size
    assert (state_after_second_batch[:min_batch_size, ...] - state_after_first_batch[:min_batch_size, ...]).sum() > 0


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
    