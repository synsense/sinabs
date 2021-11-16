import torch
from torch.onnx.symbolic_opset9 import floor, div, relu


class ThresholdSubtract(torch.autograd.Function):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    @staticmethod
    def forward(ctx, data: torch.tensor, threshold: float = 1.0, window: float = 1.0):
        """"""
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        return (data > 0) * torch.div(data, threshold, rounding_mode="trunc").float()

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (data,) = ctx.saved_tensors
        grad = ((data >= (ctx.threshold - ctx.window)).float()) / ctx.threshold
        grad_input = grad_output * grad

        return grad_input, None, None

    def symbolic(g, data: torch.tensor, threshold: float = 1.0, window: float = 1.0):
        """"""
        x = relu(g, data)
        x = div(g, x, torch.tensor(threshold))
        x = floor(g, x)
        return x


class ThresholdReset(torch.autograd.Function):
    """
    Same as `threshold_subtract`, except that the potential is reset, rather than
    subtracted. In other words, only one output spike is possible.
    Step hat gradient ___---___
    """

    @staticmethod
    def forward(ctx, data: torch.tensor, threshold: float = 1.0, window: float = 1.0):
        """"""
        ctx.save_for_backward(data)
        ctx.threshold = threshold
        ctx.window = window or threshold
        return (data >= ctx.threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (data,) = ctx.saved_tensors
        grad_input = grad_output * ((data >= (ctx.threshold - ctx.window)).float())
        return grad_input, None, None
