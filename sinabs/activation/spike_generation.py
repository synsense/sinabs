import torch
from typing import Dict, Tuple, Callable, Union, List


class BackwardClass:
    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        grad = ctx.surrogate_grad_fn(v_mem, ctx.threshold)
        grad_input = grad_output * grad
        return grad_input, None, None


class MultiSpike(BackwardClass, torch.autograd.Function):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    required_states: List[str] = ["v_mem"]

    @staticmethod
    def forward(
        ctx,
        v_mem: torch.Tensor,
        threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        """"""
        ctx.save_for_backward(v_mem.clone())
        ctx.threshold = threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = (v_mem > 0) * torch.div(
            v_mem, threshold, rounding_mode="trunc"
        ).float()
        return spikes


class SingleSpike(BackwardClass, torch.autograd.Function):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    required_states: List[str] = ["v_mem"]

    @staticmethod
    def forward(
        ctx,
        v_mem: torch.Tensor,
        threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        """"""
        ctx.save_for_backward(v_mem.clone())
        ctx.threshold = threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = (v_mem - threshold >= 0).float()
        return spikes
