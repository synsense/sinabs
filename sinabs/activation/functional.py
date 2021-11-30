import torch
from typing import Dict, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict


class IntegerDivide(torch.autograd.Function):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    @staticmethod
    def forward(ctx, states: Dict[str, torch.Tensor], threshold: float, options: dict = None):
        """"""
        ctx.save_for_backward(states['v_mem'].clone())
        ctx.threshold = threshold
        ctx.options = options
        spikes = (states['v_mem'] > 0) * torch.div(states['v_mem'], threshold, rounding_mode="trunc").float()
        states['v_mem'] = states['v_mem'] - spikes
        return spikes, states

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        
        window = ctx.options['window']
        grad = ((v_mem >= (ctx.threshold - window)).float()) / ctx.threshold
        grad_input = grad_output * grad
        return grad_input, None, None


class SubtractThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states: Dict[str, torch.Tensor], threshold: float, options: dict = None):
        """"""
        ctx.save_for_backward(states['v_mem'].clone())
        ctx.threshold = threshold
        ctx.options = options
        spikes = (states['v_mem'] - threshold > 0).float()
        states['v_mem'] = states['v_mem'] - spikes
        return spikes, states

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        window = ctx.options['window']
        grad = ((v_mem >= (ctx.threshold - window)).float()) / ctx.threshold
        grad_input = grad_output * grad
        return grad_input, None, None


class ResetMembrane(torch.autograd.Function):
    @staticmethod
    def forward(ctx, states: Dict[str, torch.Tensor], threshold: float, options: dict = None):
        """"""
        ctx.save_for_backward(states['v_mem'].clone())
        ctx.threshold = threshold
        ctx.options = options
        spikes = (states['v_mem'] - threshold > 0).float()
        states['v_mem'] = 0
        return spikes, states

    @staticmethod
    def backward(ctx, grad_output: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        window = ctx.options['window']
        grad = ((v_mem >= (ctx.threshold - window)).float()) / ctx.threshold
        grad_input = grad_output * grad
        return grad_input, None, None


@dataclass
class ActivationFunction:
    """
    Wrapper class for torch.autograd.Function with custom forward and backward passes.
    The goal is to provide flexibility in terms of spike mechanism and how to replace
    the non-differential Dirac delta activation by means of a surrogate gradient
    function.
    
    Parameters:
        spike_threshold: float
            Spikes are emitted if v_mem is above that threshold.
        spike_mechanism: str
            'divide': integer divide the membrane potential for the possibility to emit
            multiple spikes per time step. 'subtract': if v_mem is above threshold, subtract
            the threshold and emit a maximum of one spike per time step. 'reset': if v_mem is
            above threshold, emit one spike per time step and reset v_mem to zero.
        surrogate_grad: str
            Choose how to treat the spiking non-linearity during the backward pass. This is
            a function of membrane potential and 'smoothes' the activation. Options are:
            'boxcar' (default), 'exp', 'multi-gauss'.
        surrogate_options: dict
            A dictionary of options that might change depending on the surrogate gradient 
            function used.
    """
    
    spike_threshold: float = 1.
    spike_mechanism: Callable = IntegerDivide
#     surrogate_grad: Callable = BoxcarFunction,
    surrogate_options: dict = field(default_factory=lambda: {'window': 1.})

    def __call__(self, states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Takes in neuron states and returns a tuple of (spikes, new states). """
        return self.spike_mechanism.apply(states, self.spike_threshold, self.surrogate_options)
