import torch
from typing import Dict, Tuple, Callable
from dataclasses import dataclass, field
from .reset_mechanism import MembraneSubtract
from .spike_generation import MultiSpike
from .surrogate_gradient_fn import Heaviside


@dataclass
class ActivationFunction:
    """
    Wrapper class for torch.autograd.Function with custom forward and backward passes.
    The goal is to provide flexibility in terms of spike mechanism and how to replace
    the non-differential Dirac delta activation by means of a surrogate gradient
    function. The default is an activation function with threshold = 1., multiple spikes
    per time step, a membrane subtract function and a Heaviside surrogate gradient. 
    
    Parameters:
        spike_threshold: float
            Spikes are emitted if v_mem is above that threshold.
        spike_fn: torch.autograd.Function
            Choose a Sinabs or custom torch.autograd.Function that takes a dict of states, 
            a spike threshold and a surrogate gradient function and returns spikes. Be aware 
            that the class itself is passed here (because torch.autograd methods are static)
            rather than an object instance.
        reset_fn: Callable
            A function that defines how the membrane potential is reset after a spike.
        surrogate_grad_fn: Callable
            Choose how to define gradients for the spiking non-linearity during the 
            backward pass. This is a function of membrane potential.
    """
    
    spike_threshold: float = 1.
    spike_fn: Callable = MultiSpike
    reset_fn: Callable = MembraneSubtract()
    surrogate_grad_fn: Callable = Heaviside()

    def __call__(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Takes in neuron states and returns a tuple of (spikes, new states). """
        spikes = self.spike_fn.apply(state, 
                                     self.spike_threshold, 
                                     self.surrogate_grad_fn)
        state = self.reset_fn(spikes, state, self.spike_threshold)
        return spikes, state


@dataclass
class ALIFActivationFunction:
    """
    Modifies the class to pass the neuron layer's threshold state.
    
    Parameters:
        spike_fn: torch.autograd.Function
            Choose a Sinabs or custom torch.autograd.Function that takes a dict of states, 
            a spike threshold and a surrogate gradient function and returns spikes. Be aware 
            that the class itself is passed here (because torch.autograd methods are static)
            rather than an object instance.
        reset_fn: Callable
            A function that defines how the membrane potential is reset after a spike.
        surrogate_grad_fn: Callable
            Choose how to define gradients for the spiking non-linearity during the 
            backward pass. This is a function of membrane potential.
    """

    spike_fn: Callable = MultiSpike
    reset_fn: Callable = MembraneSubtract()
    surrogate_grad_fn: Callable = Heaviside()
    
    def __call__(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Takes in neuron states and returns a tuple of (spikes, new states). """
        spikes = self.spike_fn.apply(state, 
                                     state['threshold'], 
                                     self.surrogate_grad_fn)
        state = self.reset_fn(spikes, state, state['threshold'])
        return spikes, state
