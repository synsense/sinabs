import torch
from sinabs.layers import NeuromorphicReLU


def test_forward():
    t_inp = torch.tensor([-5.5, -3.23, 2.3, 0.0])
    relu = NeuromorphicReLU()
    t_out = relu(t_inp)

    assert (t_out == torch.tensor([0.0, 0.0, 2.0, 0.0])).all()


def test_backward():
    t_inp = torch.tensor([-5.5, -3.23, 2.3, 0.1], requires_grad=True)
    relu = NeuromorphicReLU()
    t_out = relu(t_inp)
    z = 2 * t_out.sum()
    z.backward()
    grad = t_inp.grad

    assert (grad == torch.tensor([0.0, 0.0, 2.0, 2.0])).all()
