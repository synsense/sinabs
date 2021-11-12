from typing import Optional, Union

import torch

from .functional import ThresholdSubtract, ThresholdReset
from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class

window = 1.0

__all__ = ["IAF", "IAFSqueeze"]

_backends_iaf = dict()


class IAF(SpikingLayer):
    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Union[float, None] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset=False,
        window: float = 1,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a Integrate and Fire neuron with learning enabled.
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        Parameters
        ----------
        threshold : float
            Spiking threshold of the neuron.
        threshold_low : float or None
            Lower bound for membrane potential.
        membrane_subtract : float or None
            The amount to subtract from the membrane potential upon spiking.
            Default is equal to threshold. Ignored if membrane_reset is set.
        membrane_reset : bool
            If True, reset the membrane to 0 on spiking.
        window : float
            Distance between step of Heaviside surrogate gradient and threshold.
            (Relative to size of threshold)
        """
        super().__init__(
            *args,
            **kwargs,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )
        self.reset_function = ThresholdReset if membrane_reset else ThresholdSubtract
        self.learning_window = threshold * window

    def forward(self, input_spikes: torch.Tensor):
        """
        Forward pass with given data.

        Parameters
        ----------
        input_spikes : torch.Tensor
            Data to be processed. Expected shape: (batch, time, ...)

        Returns
        -------
        torch.Tensor
            Output data. Same shape as `input_spikes`.
        """

        # Ensure the neuron state are initialized
        shape_notime = (input_spikes.shape[0], *input_spikes.shape[2:])
        if self.state.shape != shape_notime:
            self.reset_states(shape=shape_notime, randomize=False)

        # Determine no. of time steps from input
        time_steps = input_spikes.shape[1]

        # Local variables
        threshold = self.threshold
        threshold_low = self.threshold_low

        state = self.state
        activations = self.activations
        spikes = []
        for iCurrentTimeStep in range(time_steps):

            # update neuron states (membrane potentials)
            if self.membrane_reset:
                # sum the previous state only where there were no spikes
                state = input_spikes[:, iCurrentTimeStep] + state * (activations == 0.0)
            else:
                # subtract a number of membrane_subtract's as there are spikes
                state = (
                    input_spikes[:, iCurrentTimeStep]
                    + state
                    - activations * self.membrane_subtract
                )
            if threshold_low is not None:
                # This is equivalent to functional.threshold. non zero threshold is not supported for onnx
                state = torch.nn.functional.relu(state - threshold_low) + threshold_low

            # generate spikes
            activations = self.reset_function.apply(
                state, threshold, self.learning_window
            )
            spikes.append(activations)

        self.state = state
        self.tw = time_steps
        self.activations = activations
        all_spikes = torch.stack(spikes).transpose(0, 1)
        self.spikes_number = all_spikes.abs().sum()

        return all_spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(window=self.learning_window / self.threshold)

        return param_dict


# - Subclass to IAF, that accepts and returns data with batch and time dimensions squeezed.
IAFSqueeze = squeeze_class(IAF)
