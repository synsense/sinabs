import torch
from typing import Dict, Tuple, Callable


class BackwardClass:
    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        grad = ctx.surrogate_grad_fn(v_mem, ctx.threshold) 
        grad_input = grad_output * grad
        return grad_input, None, None


class MultiSpike(torch.autograd.Function, BackwardClass):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    @staticmethod
    def forward(ctx, state: Dict[str, torch.Tensor], threshold: float, surrogate_grad_fn: Callable):
        """"""
        ctx.save_for_backward(state['v_mem'].clone())
        ctx.threshold = threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = (state['v_mem'] > 0) * torch.div(state['v_mem'], threshold, rounding_mode="trunc").float()
        return spikes


class SingleSpike(torch.autograd.Function, BackwardClass):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    @staticmethod
    def forward(ctx, state: Dict[str, torch.Tensor], threshold: float, surrogate_grad_fn: Callable):
        """"""
        ctx.save_for_backward(state['v_mem'].clone())
        ctx.threshold = threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = (state['v_mem'] - threshold > 0).float()
        return spikes
