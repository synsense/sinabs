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
        # pre-compute leakage factor for single time step
        self.alpha = torch.exp(-1.0/torch.tensor(tau_mem))

    def detect_spikes(self, threshold):
        """
        Given the parameters, compute the spikes that will be generated.
        NOTE: This method only computes the spikes but does not reset the membrane potential.
        """
        # generate spikes
        if self.membrane_reset:
            self.activations = threshold_reset(self.state, threshold, self.threshold * window)
        else:
            self.activations = threshold_subtract(self.state, threshold, self.threshold * window)

    def update_state_after_spike(self):
        if self.membrane_reset:
            # sum the previous state only where there were no spikes
            self.state = self.state * (self.activations == 0.0)
        else:
            # subtract a number of membrane_subtract's as there are spikes
            self.state = self.state - self.activations * self.membrane_subtract

    def add_input(self, input_data):
        self.state = self.state + input_data

    def forward(self, input_current: torch.Tensor):
        """
        Forward pass with given data.

        Parameters
        ----------
        input_current : torch.Tensor
            Data to be processed. Expected shape: (batch, time, ...)

        Returns
        -------
        torch.Tensor
            Output data. Same shape as `input_spikes`.
        """
        # Ensure the neuron state are initialized
        shape_notime = (input_current.shape[0], *input_current.shape[2:])
        if self.state.shape != shape_notime:
            self.reset_states(shape=shape_notime, randomize=False)

        time_steps = input_current.shape[1]
        output_spikes = torch.zeros_like(input_current)
        
        for step in range(time_steps):
            # generate spikes
            self.detect_spikes(threshold=self.threshold)
            output_spikes[:, step] = self.activations

            # Reset membrane potential for neurons that spiked
            self.update_state_after_spike()

            # Decay the membrane potential
            self.state *= self.alpha

            # Add the input currents to membrane potential
            self.add_input(input_current[:, step])

            # Membrane potential lower bound
            if self.threshold_low is not None:
                self.state = torch.clamp(self.state, min=self.threshold_low)

        self.tw = time_steps
        self.spikes_number = output_spikes.abs().sum()

        return output_spikes


LIFSqueeze = squeeze_class(LIF)
