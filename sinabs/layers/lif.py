from typing import Optional, Union
import torch
from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class
from .functional import threshold_subtract, threshold_reset


__all__ = ["LIF", "LIFSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class LIF(SpikingLayer):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
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
            *args,
            **kwargs,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )
        self.tau_mem = tau_mem

    def detect_spikes(self):
        """
        Given the parameters, compute the spikes that will be generated.
        NOTE: This method only computes the spikes but does not reset the membrate potential.
        """
        # generate spikes
        if self.membrane_reset:
            self.activations = threshold_reset(self.state, self.threshold, self.threshold * window)
        else:
            self.activations = threshold_subtract(self.state, self.threshold, self.threshold * window)

    def update_state_after_spike(self):
        if self.membrane_reset:
            # sum the previous state only where there were no spikes
            self.state = self.state * (self.activations == 0.0)
        else:
            # subtract a number of membrane_subtract's as there are spikes
            self.state = self.state - self.activations * self.membrane_subtract

    def do_leak(self):
        alpha = torch.exp(- 1.0/self.tau_mem)
        return alpha * self.state

    def add_input(self, input_data):
        self.state = self.state + input_data # Add input

    def forward(self, input_current: torch.Tensor):
        # Ensure the neuron state are initialized
        shape_notime = (input_current.shape[0], *input_current.shape[2:])
        if self.state.shape != shape_notime:
            self.reset_states(shape=shape_notime, randomize=False)

        # Determine no. of time steps from input
        time_steps = input_current.shape[1]

        spikes = []

        for iCurrentTimeStep in range(time_steps):
            # generate spikes
            self.detect_spikes()
            spikes.append(self.activations)

            # Reset membrane potential
            self.update_state_after_spike()

            # Do leak
            self.do_leak()

            # Add input
            self.add_input(input_current[:, iCurrentTimeStep])  # Add input

            # Membrane potential lower bound
            if self.threshold_low is not None:
                # This is equivalent to functional.threshold. non zero threshold is not supported for onnx
                self.state = torch.nn.functional.relu(self.state - self.threshold_low) + self.threshold_low

        self.tw = time_steps
        all_spikes = torch.stack(spikes).transpose(0, 1)
        self.spikes_number = all_spikes.abs().sum()

        return all_spikes


LIFSqueeze = squeeze_class(LIF)
