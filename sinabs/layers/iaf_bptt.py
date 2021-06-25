from typing import Optional, Union

import torch

from .functional import threshold_subtract, threshold_reset
from .spiking_layer import SpikingLayer
from .pack_dims import squeeze_class

window = 1.0

__all__ = ["IAF", "IAFSqueeze"]


class IAF(SpikingLayer):
    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Union[float, None] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset=False,
    ):
        """
        Pytorch implementation of a Integrate and Fire neuron with learning enabled.
        This class is the base class for any layer that need to implement integrate-and-fire operations.

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
        super().__init__()
        # Initialize neuron states
        self.threshold = threshold
        self.threshold_low = threshold_low
        self._membrane_subtract = membrane_subtract
        self.membrane_reset = membrane_reset

        # Blank parameter place holders
        self.register_buffer("state", torch.zeros(1))
        self.register_buffer("activations", torch.zeros(1))
        self.spikes_number = None

    @property
    def membrane_subtract(self):
        if self._membrane_subtract is not None:
            return self._membrane_subtract
        else:
            return self.threshold

    @membrane_subtract.setter
    def membrane_subtract(self, new_val):
        self._membrane_subtract = new_val

    def reset_states(self, shape=None, randomize=False):
        """
        Reset the state of all neurons in this layer
        """
        device = self.state.device
        if shape is None:
            shape = self.state.shape

        if randomize:
            # State between lower and upper threshold
            low = self.threshold_low or -self.threshold
            width = self.threshold - low
            self.state = torch.rand(shape, device=device) * width + low
            self.activations = torch.zeros(shape, device=self.activations.device)
        else:
            self.state = torch.zeros(shape, device=self.state.device)
            self.activations = torch.zeros(shape, device=self.activations.device)

    def detach_state_grad(self):
        """
        Remove gradients from stored states and activations. This is helpful when
        state should be maintained between `forward` calls but backpropagation of
        gradients should not extend beyond single `forward` call.
        """
        self.state = self.state.data
        self.activations = self.activations.data

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
            if self.membrane_reset:
                activations = threshold_reset(state, threshold, threshold * window)
            else:
                activations = threshold_subtract(state, threshold, threshold * window)
            spikes.append(activations)

        self.state = state
        self.tw = time_steps
        self.activations = activations
        all_spikes = torch.stack(spikes).transpose(0, 1)
        self.spikes_number = all_spikes.abs().sum()

        return all_spikes

    def get_output_shape(self, in_shape):
        """
        Returns the output shape for passthrough implementation

        :param in_shape:
        :return: out_shape
        """
        return in_shape

    def __deepcopy__(self, memo=None):
        # TODO: What is `memo`?

        other = self.__class__(**self._param_dict)

        other.state = self.state.detach().clone()
        other.activations = self.activations.detach().clone()

        return other

    @property
    def _param_dict(self) -> dict:
        """
        Dict of all parameters relevant for creating a new instance with same
        parameters as `self`
        """
        param_dict = super()._param_dict()
        param_dict.update(
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self._membrane_subtract,
            membrane_reset=self.membrane_reset,
        )
        return param_dict


# - Subclass to IAF, that accepts and returns data with batch and time dimensions squeezed.
IAFSqueeze = squeeze_class(IAF)
