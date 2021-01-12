import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from .functional import threshold_subtract, threshold_reset

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

window = 1.0


class SpikingLayer(nn.Module):
    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = None,
        batch_size: Optional[int] = None,
        membrane_reset=False,
    ):
        """
        Pytorch implementation of a spiking neuron with learning enabled.
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        :param threshold: Spiking threshold of the neuron.
        :param threshold_low: Lower bound for membrane potential.
        :param membrane_subtract: The amount to subtract from the membrane potential upon spiking. \
        Default is equal to threshold. Ignored if membrane_reset is set.
        :param negative_spikes: Implement a linear transfer function through negative spiking. \
        Ignored if membrane_reset is set.
        :param batch_size: The batch size. Needed to distinguish between timesteps and batch dimension.
        :param membrane_reset: bool, if True, reset the membrane to 0 on spiking.
        :param layer_name: The name of the layer.
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
        self.batch_size = batch_size

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

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method needs to be overridden/defined by the child class
        Default implementation is pass through

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        return input_spikes

    def forward(self, binary_input: torch.Tensor):

        # Compute the synaptic current
        syn_out: torch.Tensor = self.synaptic_output(binary_input)

        # Reshape data to appropriate dimensions
        if self.batch_size:
            syn_out = syn_out.reshape((self.batch_size, -1, *syn_out.shape[1:])).transpose(0, 1)
        # Ensure the neuron state are initialized
        try:
            assert self.state.shape == syn_out.shape[1:]
        except AssertionError:
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

        if self.batch_size:
            all_spikes = all_spikes.transpose(0, 1).reshape((-1, *all_spikes.shape[2:]))
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
