import torch
import sinabs.layers as sl


def test_flatten_time():
    data = torch.rand((5, 100, 2, 10, 10))
    flatten = sl.FlattenTime()
    output = flatten(data)

    assert output.shape == (500, 2, 10, 10)


def test_unflatten_time():
    data = torch.rand((500, 2, 10, 10))
    unflatten = sl.UnflattenTime(batch_size=5)
    output = unflatten(data)

    assert output.shape == (5, 100, 2, 10, 10)
