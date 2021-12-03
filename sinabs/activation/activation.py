import torch
from typing import Dict, Tuple, Callable
from dataclasses import dataclass, field
from .reset_mechanism import membrane_subtract
from .spike_generation import multi_spike
from .surrogate_gradient_fn import Heaviside


class Activation(torch.autograd.Function):
    """
    PyTorch-compatible function that returns the number of spikes emitted,
    given a membrane potential value and in a "threshold subtracting" regime.
    In other words, the integer division of the input by the threshold is returned.
    In the backward pass, the gradient is zero if the membrane is at least
    `threshold - window`, and is passed through otherwise.
    """

    @staticmethod
    def forward(ctx, 
                state: Dict[str, torch.Tensor], 
                threshold: float, 
                spike_fn: Callable, 
                reset_fn: Callable, 
                surrogate_grad_fn: Callable
               ):
        """"""
        ctx.save_for_backward(state['v_mem'].clone())
        ctx.threshold = threshold
        ctx.surrogate_grad_fn = surrogate_grad_fn
        spikes = spike_fn(state, threshold)
        state = reset_fn(spikes, state, threshold)
        return spikes, state

    @staticmethod
    def backward(ctx, grad_output: torch.tensor, grad_output_states: torch.tensor):
        """"""
        (v_mem,) = ctx.saved_tensors
        grad = ctx.surrogate_grad_fn(v_mem, ctx.threshold) 
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
        spike_fn: Callable
            Choose a Sinabs or custom spike function that takes a dict of states and spike
            threshold and returns spikes.
        surrogate_grad: str
            Choose how to treat the spiking non-linearity during the backward pass. This is
            a function of membrane potential and 'smoothes' the activation. Options are:
            'boxcar' (default), 'exp', 'multi-gauss'.
        surrogate_options: dict
            A dictionary of options that might change depending on the surrogate gradient 
            function used.
    """
    
    spike_threshold: float = 1.
    spike_fn: Callable = multi_spike
    reset_fn: Callable = membrane_subtract
    surrogate_grad_fn: Callable = Heaviside()

    def __call__(self, states: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Takes in neuron states and returns a tuple of (spikes, new states). """
        return Activation.apply(states, 
                                self.spike_threshold, 
                                self.spike_fn,
                                self.reset_fn,
                                self.surrogate_grad_fn)
