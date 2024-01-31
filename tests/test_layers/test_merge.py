import torch

import sinabs.layers as sl


def test_morph_same_size():
    data1 = (torch.rand((100, 1, 20, 20)) > 0.5).float()
    data2 = (torch.rand((100, 1, 20, 20)) > 0.5).float()

    merge = sl.Merge()
    out = merge(data1, data2)
    assert out.shape == (100, 1, 20, 20)


def test_morph_different_size():
    data1 = (torch.rand((100, 1, 5, 6)) > 0.5).float()
    data2 = (torch.rand((100, 10, 5, 5)) > 0.5).float()

    merge = sl.Merge()
    out = merge(data1, data2)

    assert out.shape == (100, 10, 5, 6)
    assert out.sum() == data1.sum() + data2.sum()
