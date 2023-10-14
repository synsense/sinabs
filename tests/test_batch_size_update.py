import pytest
import torch


@pytest.mark.parametrize(
    "first_batch_size,second_batch_size", [(2, 5), (5, 7), (2, 11)]
)
def test_batch_size_mismatch(first_batch_size, second_batch_size):
    import sinabs
    import sinabs.layers as sl

    mod = sl.IAFSqueeze(batch_size=first_batch_size)

    time_len = 13

    first_data = torch.rand((first_batch_size, 13, 57)).reshape(
        (first_batch_size * time_len, -1)
    )
    out = mod(first_data)

    second_data = torch.rand((second_batch_size, 13, 57)).reshape(
        second_batch_size * time_len, -1
    )
    # Check that it fails the first time without batch change
    with pytest.raises(RuntimeError):
        out = mod(second_data)

    sinabs.set_batch_size(mod, second_batch_size)
    out = mod(second_data)
