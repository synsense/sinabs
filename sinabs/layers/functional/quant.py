import torch.autograd


class Quantize(torch.autograd.Function):
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


quantize = Quantize().apply
stochastic_rounding = StochasticRounding().apply
