from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MembraneReset:
    """
    Reset the membrane potential v_mem to a given value
    after it spiked.
    
    Parameters:
        reset_value: fixed value that a neuron should be reset to. Defaults to zero.
        
    Example:
        >>> activation_fn = sinabs.activation.ActivationFunction(
        >>>                     reset_fn=MembraneReset(reset_value=0.)
        >>>                     )
        >>> layer = sinabs.layers.LIF(activation_fn=activation_fn, ...)
    """
    
    reset_value: float = 0.

    def __call__(self, spikes, state, threshold):
        state['v_mem'] = self.reset_value
        return state


@dataclass
class MembraneSubtract:
    """
    Subtract the spiking threshold from the membrane potential
    for every neuron that spiked.
    
    Parameters:
        subtract_value: optional value that will be subtraced from
                        v_mem if it spiked. Defaults to spiking threshold if None.

    Example:
        >>> activation_fn = sinabs.activation.ActivationFunction(
        >>>                     reset_fn=MembraneSubtract(subtract_value=None)
        >>>                     )
        >>> layer = sinabs.layers.LIF(activation_fn=activation_fn, ...)
    """

    subtract_value: Optional[float] = None
    
    def __call__(self, spikes, state, threshold):
        if self.subtract_value:
            state['v_mem'] = state['v_mem'] - spikes * self.subtract_value        
        else:
            state['v_mem'] = state['v_mem'] - spikes * threshold
        return state
