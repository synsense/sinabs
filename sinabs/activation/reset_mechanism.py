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

    reset_value: float = 0.0

    def __call__(self, spikes, state, threshold):
        new_state = state.copy()
        new_state["v_mem"] = (
            new_state["v_mem"] * (spikes == 0).float() + self.reset_value
        )
        return new_state


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
        new_state = state.copy()
        if self.subtract_value:
            new_state["v_mem"] = new_state["v_mem"] - spikes * self.subtract_value
        else:
            new_state["v_mem"] = new_state["v_mem"] - spikes * threshold
        return new_state
