from typing import Optional, Union
import torch
from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class
from .functional import ThresholdSubtract, ThresholdReset


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

        .. math ::
            \\tau_{mem} \\dot{V}_{mem} = -V_{mem} + \\sum z(t)

            \\text{if } V_m(t) = V_{th} \\text{, then } V_{m} \\rightarrow V_{reset}

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
        self.tau_mem = torch.tensor(tau_mem)
        self.reset_function = ThresholdReset if membrane_reset else ThresholdSubtract

    def check_states(self, input_current):
        shape_without_time = (input_current.shape[0], *input_current.shape[2:])
        if self.state.shape != shape_without_time:
            self.reset_states(shape=shape_without_time, randomize=False)

    def detect_spikes(self):
        """
        Given the parameters, compute the spikes that will be generated.
        NOTE: This method only computes the spikes but does not reset the membrane potential.
        """
        self.activations = self.reset_function.apply(self.state,
                                                     self.threshold,
                                                     self.threshold * window)

    def update_state_after_spike(self):
        if self.membrane_reset:
            # sum the previous state only where there were no spikes
            self.state = self.state * (self.activations == 0.0)
        else:
            # subtract a number of membrane_subtract's as there are spikes
            self.state = self.state - self.activations * self.membrane_subtract

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
        self.check_states(input_current)

        time_steps = input_current.shape[1]
        output_spikes = torch.zeros_like(input_current)

        for step in range(time_steps):
            # generate spikes
            self.detect_spikes()
            output_spikes[:, step] = self.activations

            # Reset membrane potential for neurons that spiked
            self.update_state_after_spike()

            # Decay the membrane potential
            alpha = torch.exp(-1.0/self.tau_mem)
            self.state *= alpha

            # Add the input currents to membrane potential state
            self.state += input_current[:, step]

            # Clip membrane potential that is too low
            if self.threshold_low: self.state = torch.clamp(self.state, min=self.threshold_low)

        self.tw = time_steps
        self.spikes_number = output_spikes.abs().sum()
        return output_spikes


LIFSqueeze = squeeze_class(LIF)
