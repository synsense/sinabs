import torch


class ActivationFunction:
    def __init__(self,
                 spike_threshold: float = 1,
                 spike_mechanism: str = 'divide',
                 surrogate_grad: str = 'boxcar',
                 surrogate_window: float = 1.,
                ):
        self.spike_treshold = spike_threshold
        self.spike_mechanism = spike_mechanism
        self.surrogate_window = surrogate_window
        
    def __call__(self, states):
        if self.spike_mechanism == 'divide':
            return ThresholdDivide.apply(states['v_mem'], self.spike_treshold, self.surrogate_window)
        else:
            raise NotImplementedError


class ThresholdDivide(torch.autograd.Function):
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
