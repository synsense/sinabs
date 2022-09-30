import torch
import sinabs.layers as sl
import torch.nn as nn


def test_repeat():
    data = torch.rand((5, 100, 2, 10, 10))
    conv = nn.Conv2d(2, 4, 10)
    repeated_conv = sl.Repeat(conv)
    output = repeated_conv(data)
    alt_output = conv(data.flatten(0, 1)).unflatten(0, (5, 100))

    torch.testing.assert_allclose(output, alt_output)


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
