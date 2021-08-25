from typing import Optional, Union
import torch
from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class
from .functional import threshold_subtract, threshold_reset
from .lif import LIF


__all__ = ["ALIF", "ALIFSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class ALIF(LIF):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_thresh: Union[float, torch.Tensor],
        delta_thresh: Union[float, torch.Tensor] = 0.1,
        threshold: Union[float, torch.Tensor] = 1.0,
        threshold_low: Union[float, None] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a Leaky Integrate and Fire neuron with learning enabled.

        Parameters
        ----------
        threshold: float
            Spiking threshold of the neuron.
        threshold_low: float or None
            Lower bound for membrane potential.
        membrane_subtract: float or None
            The amount to subtract from the membrane potential upon spiking.
            Default is equal to threshold. Ignored if membrane_reset is set.
        membrane_reset: bool
            If True, reset the membrane to 0 on spiking.
        """
        super().__init__(
            tau_mem=tau_mem,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            *args,
            **kwargs,
        )
        # pre-compute leakage factor for single time step
        self.alpha = torch.exp(-1.0/torch.tensor(tau_mem))
        self.beta = torch.exp(-1.0/torch.tensor(tau_thresh))
        self.delta_thresh = delta_thresh
        self.thresh_state = None

    def check_states(self, input_current):
        shape_notime = (input_current.shape[0], *input_current.shape[2:])
        if self.state.shape != shape_notime:
            self.reset_states(shape=shape_notime, randomize=False)
        if self.thresh_state == None:
            self.thresh_state = torch.ones_like(input_current[:,0]) * self.threshold

    def detect_spikes(self):
        """
        Given the parameters, compute the spikes that will be generated.
        NOTE: This method only computes the spikes but does not reset the membrane potential.
        """
        # generate spikes
        if self.membrane_reset:
            self.activations = threshold_reset(self.state, self.thresh_state, self.threshold * window)
        else:
            self.activations = threshold_subtract(self.state, self.thresh_state, self.threshold * window)

    def update_state_after_spike(self):
        if self.membrane_reset:
            # sum the previous state only where there were no spikes
            self.state = self.state * (self.activations == 0.0)
        else:
            # subtract a number of membrane_subtract's as there are spikes
            self.state = self.state - self.activations * self.membrane_subtract
        # Increase spike thresholds if neuron spiked
        self.adapt_threshold_state(self.activations)

    def adapt_threshold_state(self, output_spikes):
        # Decay the spike threshold
        self.thresh_state *= self.beta
        self.thresh_state += output_spikes * self.delta_thresh

ALIFSqueeze = squeeze_class(ALIF)
