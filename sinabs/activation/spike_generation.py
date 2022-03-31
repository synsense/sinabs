from dataclasses import dataclass
import torch
from typing import Optional, Callable, Union, List


class BackwardClass:
    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        grad = ctx.surrogate_grad_fn(v_mem, ctx.spike_threshold)
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
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        """"""
        ctx.save_for_backward(v_mem.clone())
        ctx.spike_threshold = spike_threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = (v_mem > 0) * torch.div(
            v_mem, spike_threshold, rounding_mode="trunc"
        ).float()
        return spikes


class MaxSpikeInner(BackwardClass, torch.autograd.Function):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    Other than MultiSpike, the number of spikes emitted in one time step is limited.
    Equivalent to SingleSpike for max_num_spikes_per_bin=1 and to MultiSpike for
    max_num_spikes_per_bin=None.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    required_states: List[str] = ["v_mem", "max_num_spikes_per_bin"]

    @staticmethod
    def forward(
        ctx,
        v_mem: torch.Tensor,
        max_num_spikes_per_bin: Optional[int],
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        """"""
        ctx.save_for_backward(v_mem.clone())
        ctx.spike_threshold = spike_threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = (v_mem > 0) * torch.div(
            v_mem, spike_threshold, rounding_mode="trunc"
        ).float()
        if max_num_spikes_per_bin is not None:
            spikes = torch.clamp(spikes, max=max_num_spikes_per_bin)
        return spikes


@dataclass
class MaxSpike:
    """
    Wrapper for MaxSpikeInner that does not require passing max_num_spikes_per_bin
    when calling apply but only at instantiation.
    """

    max_num_spikes_per_bin: Optional[int] = None

    def apply(
        self,
        v_mem: torch.Tensor,
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        return MaxSpikeInner.apply(
            v_mem, self.max_num_spikes_per_bin, spike_threshold, surrogate_grad_fn
        )

    @property
    def required_states(self):
        return ["v_mem"]


class SingleSpike(BackwardClass, torch.autograd.Function):
    """
    PyTorch-compatible function that returns a single spike per time step.
    In the backward pass, the gradient is zero if the membrane is at least
    `spike_threshold - window`, and is passed through otherwise.
    """

    required_states: List[str] = ["v_mem"]

    @staticmethod
    def forward(
        ctx,
        v_mem: torch.Tensor,
        spike_threshold: Union[float, torch.Tensor],
        surrogate_grad_fn: Callable,
    ):
        """"""
        ctx.save_for_backward(v_mem.clone())
        ctx.spike_threshold = spike_threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = (v_mem - spike_threshold >= 0).float()
        return spikes
