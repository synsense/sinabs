from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import torch


class BackwardClass:
    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        grad = ctx.surrogate_grad_fn(v_mem, ctx.spike_threshold)
        grad_input = grad_output * grad
        return grad_input, None, None, None


class MultiSpike(BackwardClass, torch.autograd.Function):
    """Autograd function that returns membrane potential integer-divided by spike threshold. Do not
    instantiate this class when passing as spike_fn (see example). Can be combined with different
    surrogate gradient functions.

    Example:
        >>> layer = sinabs.layers.LIF(spike_fn=MultiSpike, ...)
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
    """Autograd function that returns membrane potential divided by spike threshold for a maximum
    number of spikes per time step.

    Can be combined with different surrogate gradient functions.
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
    """Wrapper for MaxSpikeInner autograd function. This class needs to be instantiated when used
    as spike_fn. Notice the difference in example to Single/MultiSpike.

    Example:
        >>> layer = sinabs.layers.LIF(spike_fn=MaxSpike(max_num_spikes_per_bin=10), ...)
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
    """Autograd function that returns membrane potential divided by spike threshold for a maximum
    of one spike per time step. Do not instantiate this class when passing as spike_fn (see
    example). Can be combined with different surrogate gradient functions.

    Example:
        >>> layer = sinabs.layers.LIF(spike_fn=SingleSpike, ...)
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
