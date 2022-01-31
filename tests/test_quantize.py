def test_quantize():
    import torch
    from sinabs.layers import NeuromorphicReLU

    x = torch.rand(20, requires_grad=True)
    lyr = NeuromorphicReLU(quantize=True, fanout=1, stochastic_rounding=False)

    interm = lyr(x)
    out = 40 * interm
    err = (out.sum() - 100) ** 2
    err.backward()
    assert out.sum() == 0
    assert x.grad.sum() != 0


def test_stochastic_rounding():
    import torch
    from sinabs.layers import NeuromorphicReLU

    x = torch.rand(20, requires_grad=True)
    lyr = NeuromorphicReLU(quantize=True, fanout=1, stochastic_rounding=True)

    interm = lyr(x)
    out = 40 * interm
    err = (out.sum() - 100) ** 2
    err.backward()
    assert out.sum() > 0
    assert x.grad.sum() != 0
