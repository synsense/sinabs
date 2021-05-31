import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from .functional import threshold_subtract, threshold_reset

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

window = 1.0


class IAF(nn.Module):
    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset=False,
    ):
        """
        Pytorch implementation of a Integrate and Fire neuron with learning enabled.
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        :param threshold: Spiking threshold of the neuron.
        :param threshold_low: Lower bound for membrane potential.
        :param membrane_subtract: The amount to subtract from the membrane potential upon spiking. \
        Default is equal to threshold. Ignored if membrane_reset is set.
        :param membrane_reset: bool, if True, reset the membrane to 0 on spiking.
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
            self.state = torch.rand(shape, device=device)
            self.activations = torch.rand(shape, device=device)
        else:
            self.state = torch.zeros(shape, device=self.state.device)
            self.activations = torch.zeros(shape, device=self.activations.device)

    def forward(self, syn_out: torch.Tensor):
        # Expected input shape (time, ...)
        # Ensure the neuron state are initialized
        if self.state.shape != syn_out.shape[1:]:
            self.reset_states(shape=syn_out.shape[1:], randomize=False)

        # Determine no. of time steps from input
        time_steps = len(syn_out)

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
                state = syn_out[iCurrentTimeStep] + state * (activations == 0.)
            else:
                # subtract a number of membrane_subtract's as there are spikes
                state = syn_out[iCurrentTimeStep] + state - activations * self.membrane_subtract
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
        all_spikes = torch.stack(spikes)
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
        other = SpikingLayer(
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self._membrane_subtract,
            batch_size=self.batch_size,
            membrane_reset=self.membrane_reset,
        )

        other.state = self.state.detach().clone()
        other.activations = self.activations.detach().clone()

        return other


class SpikingLayer(IAF):
    """
    Pytorch implementation of a spiking neuron with learning enabled.
    This class is the base class for any layer that need to implement integrate-and-fire operations.

    Takes input of shape (batch*time, ...)

    Params
    ------
    batch_size: int/None
        Specify the batch size of the input data
        If None, batch_size 1 is assumed

    See :py:class:`IAF` class for other parameters of this class

    """
    def __init__(self, *args, batch_size=1, **kwargs):
        super().__init__(*args, **kwargs)
        if batch_size is None:
            self.batch_size = 1
        else:
            self.batch_size = batch_size

    def forward(self, data):
        # Expected input shape (batch*time, ...)
        out = data
        # Unsqueeze
        out = data.reshape((self.batch_size, -1, *data.shape[1:])).transpose(0, 1)
        out = super().forward(out)
        # Flatten batch, time
        out = out.transpose(0, 1).reshape((-1, *out.shape[2:]))
        return out
