import torch
import torch.nn as nn
from packaging import version

import sinabs.layers as sl


def test_repeat():
    data = torch.rand((5, 100, 2, 10, 10))
    conv = nn.Conv2d(2, 4, 10)
    repeated_conv = sl.Repeat(conv)
    output = repeated_conv(data)
    alt_output = conv(data.flatten(0, 1)).unflatten(0, (5, 100))

    if version.parse(torch.__version__) > version.parse("1.12"):
        torch.testing.assert_close(output, alt_output)
    else:
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
