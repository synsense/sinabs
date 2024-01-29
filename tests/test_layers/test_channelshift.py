import torch

from sinabs.layers.channel_shift import ChannelShift


def test_channel_shift_default():
    x = torch.rand(1, 10, 5, 5)
    cs = ChannelShift()

    out = cs(x)
    assert out.shape == x.shape


def test_channel_shift():
    num_channels = 10
    channel_shift = 14
    x = torch.rand(1, num_channels, 5, 5)
    cs = ChannelShift(channel_shift=channel_shift)

    out = cs(x)
    assert len(out.shape) == len(x.shape)
    assert out.shape[1] == num_channels + channel_shift
