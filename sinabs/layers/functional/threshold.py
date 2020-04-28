import torch
from torch.onnx.symbolic_opset9 import floor_divide

class ThresholdSubtract(torch.autograd.Function):
    """
    Subtract from membrane potential on reaching threshold
    """

    @staticmethod
    def forward(ctx, data, threshold=1, window=0.5):
        ctx.save_for_backward(data.clone())
        ctx.threshold = threshold
        ctx.window = window
        return (data >= ctx.threshold) * (data // ctx.threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (data,) = ctx.saved_tensors
        grad_input = grad_output * ((data >= (ctx.threshold - ctx.window)).float())
        return grad_input, None, None

    def symbolic(g, data, threshold=1, window=0.5):
        pos = g.op("Greater", data, torch.tensor(0))
        x = floor_divide(g, data, torch.tensor(threshold))
        return g.op("Mul", x, pos)





threshold_subtract = ThresholdSubtract().apply


class ThresholdReset(torch.autograd.Function):
    """
    Threshold check
    Step hat gradient ___---___
    """

    @staticmethod
    def forward(ctx, data, threshold=1, window=0.5):
        ctx.save_for_backward(data)
        ctx.threshold = threshold
        ctx.window = window
        return (data >= ctx.threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (data,) = ctx.saved_tensors
        grad_input = grad_output * ((data >= (ctx.threshold - ctx.window)).float())
        return grad_input, None, None


threshold_reset = ThresholdReset().apply
