from typing import Optional, Union
import torch
from .spiking_layer import SpikingLayer
from .recurrent_module import RecurrentModule
from .pack_dims import squeeze_class
from .functional import ThresholdSubtract, ThresholdReset


__all__ = ["LIF", "LIFSqueeze"]

# Learning window for surrogate gradient
window = 1.0


class LIF(SpikingLayer):
    def __init__(
        self,
        alpha_mem: Union[float, torch.Tensor],
        threshold: Union[float, torch.Tensor] = 1.,
        membrane_reset: bool = False,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a Leaky Integrate and Fire neuron layer.

        .. math ::
            V_{mem}(t) = V_{mem}(t-1)\\alpha + \\sum z(t)

            \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

        where :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

        Parameters
        ----------
        alpha_mem: float
            Membrane potential decay time constant.
        threshold: float
            Spiking threshold of the neuron, defaults to 1.
        membrane_reset: bool
            If True, reset the membrane to 0 on spiking. Otherwise, will divide the 
            activation by spiking threshold and truncate to integers. That means that
            muliple spikes can be generated within a single time step.
        threshold_low: float or None
            Lower bound for membrane potential.
        membrane_subtract: float or None
            The amount to subtract from the membrane potential upon spiking.
            Default is equal to threshold. Ignored if membrane_reset is set.
        """
        super().__init__(
            *args,
            **kwargs,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )
        self.alpha_mem = alpha_mem
        self.reset_function = ThresholdReset if membrane_reset else ThresholdSubtract

    def check_states(self, input_current):
        """ Initialise neuron membrane potential states when the first input is received. """
        shape_without_time = (input_current.shape[0], *input_current.shape[2:])
        if self.state.shape != shape_without_time:
            self.reset_states(shape=shape_without_time, randomize=False)

    def detect_spikes(self):
        """ Compute spike outputs for a single time step. This method does not reset the membrane potential. """
        self.activations = self.reset_function.apply(self.state,
                                                     self.threshold,
                                                     self.threshold * window)

    def update_state_after_spike(self):
        """ Update membrane potentials to either reset or subtract by given value after spikes occured at this time step. """
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
            self.state = self.state * self.alpha_mem

            # Add the input currents which are normalised by tau to membrane potential state
            self.state = self.state + (1-self.alpha_mem) * input_current[:, step]

            # Clip membrane potential that is too low
            if self.threshold_low: self.state = torch.clamp(self.state, min=self.threshold_low)

        self.tw = time_steps
        self.spikes_number = output_spikes.abs().sum()
        return output_spikes


class LIFRecurrent(RecurrentModule):
    def __init__(
        self,
        alpha_mem: Union[float, torch.Tensor],
        rec_connectivity: torch.nn.Module,
        threshold: Union[float, torch.Tensor] = 1.,
        membrane_reset: bool = False,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            layer = LIF(
                alpha_mem=alpha_mem,
                threshold=threshold,
                threshold_low=threshold_low,
                membrane_subtract=membrane_subtract,
                membrane_reset=membrane_reset,
                *args,
                **kwargs,
            ),
            rec_connectivity=rec_connectivity
            
        )
        


LIFSqueeze = squeeze_class(LIF)
LIFRecurrentSqueeze = squeeze_class(LIFRecurrent)
