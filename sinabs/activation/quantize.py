import torch.autograd


class Quantize(torch.autograd.Function):
    """PyTorch-compatible function that applies a floor() operation on the input, while providing a
    surrogate gradient (equivalent to that of a linear function) in the backward pass."""

    @staticmethod
    def forward(ctx, inp):
        """"""
        return inp.floor()

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        grad_input = grad_output.clone()
        return grad_input


class StochasticRounding(torch.autograd.Function):
    """PyTorch-compatible function that applies stochastic rounding. The input x.

    is quantized to ceil(x) with probability (x - floor(x)), and to floor(x)
    otherwise. The backward pass is provided as a surrogate gradient
    (equivalent to that of a linear function).
    """

    @staticmethod
    def forward(ctx, inp):
        """"""
        int_val = inp.floor()
        frac = inp - int_val
        output = int_val + (torch.rand_like(inp) < frac).float()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        grad_input = grad_output.clone()
        return grad_input
