import torch
import numpy as np
import torch.nn as nn
from .layer import Layer
from typing import Optional, Union, List, Tuple
from abc import abstractmethod

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingLayer(Layer):
    def __init__(
        self,
        input_shape: ArrayLike,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset: Optional[float] = None,
        layer_name: str = "spiking",
        negative_spikes: bool = False,
    ):
        """
        Pytorch implementation of a spiking neuron.
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        :param input_shape: Input data shape
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: Upon spiking, if the membrane potential is subtracted as opposed to reset, \
        what is the subtracted value? Defaults to threshold.
        :param membrane_reset: What is the reset membrane potential of the neuron. \
        If not None, the membrane potential is reset instead of subtracted on spiking.
        :param layer_name: Name of this layer
        :param negative_spikes: Implement a linear transfer function through negative spiking
        """
        super().__init__(input_shape=input_shape, layer_name=layer_name)
        # Initialize neuron states
        self._membrane_subtract = membrane_subtract
        self.membrane_reset = membrane_reset
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.negative_spikes = negative_spikes

        # Blank parameter place holders
        self.spikes_number = None
        self.state = None

    @property
    def threshold_low(self):
        return self._threshold_low

    @threshold_low.setter
    def threshold_low(self, new_threshold_low):
        self._threshold_low = new_threshold_low
        if new_threshold_low is None:
            try:
                del self.thresh_lower
            except AttributeError:
                pass
        else:
            # Relu on the layer
            self.thresh_lower = nn.Threshold(new_threshold_low, new_threshold_low)

    def reset_states(self):
        """
        Reset the state of all neurons in this layer
        """
        if self.state is None:
            return
        else:
            self.state.zero_()

    @abstractmethod
    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method needs to be overridden/defined by the child class

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        pass

    def forward(self, binary_input: torch.Tensor):
        # Determine no. of time steps from input
        neg_spikes = self.negative_spikes

        # Compute the synaptic current
        syn_out: torch.Tensor = self.synaptic_output(binary_input)
        time_steps = len(syn_out)

        # Local variables
        membrane_subtract = self.membrane_subtract
        threshold = self.threshold
        threshold_low = self.threshold_low
        membrane_reset = self.membrane_reset

        # Create a vector to hold all output spikes
        spikes = syn_out.new_zeros(time_steps, *syn_out.shape[1:])

        # Initialize state as required
        if self.state is None:
            self.state = syn_out.new_zeros(syn_out.shape[1:])
        elif self.state.device != syn_out.device:
            # print(f"Device type state: {self.state.device}, syn_out: {syn_out.device} ")
            self.state = self.state.to(syn_out.device)

        state = self.state
        self.spikes_number = 0

        # Loop over time steps
        for iCurrentTimeStep in range(time_steps):
            state = state + syn_out[iCurrentTimeStep]
            # - Reset or subtract from membrane state after spikes
            if membrane_reset is None:
                if not neg_spikes:
                    # Calculate number of spikes to be generated
                    n_thresh_crossings = (
                        (state - threshold) / membrane_subtract
                    ).int() + 1
                    spikes[iCurrentTimeStep] = (
                        state >= threshold
                    ).int() * n_thresh_crossings
                else:
                    n_thresh_crossings = (
                        (state.abs() - threshold) / membrane_subtract
                    ).floor().int() + 1
                    spikes[iCurrentTimeStep] = state.sign().int() * n_thresh_crossings

                # - Subtract from states
                state -= membrane_subtract * spikes[iCurrentTimeStep].float()
            else:
                if not neg_spikes:
                    # - Check threshold crossings for spikes
                    spike_record = state >= threshold
                    # - Add to spike counter
                    spikes[iCurrentTimeStep] = spike_record
                else:
                    # this was not tested
                    # - Check threshold crossings for spikes
                    spike_record = state.abs() >= threshold
                    # - Add to spike counter
                    spikes[iCurrentTimeStep] = spike_record * state.sign().int()

                # - Reset neuron states
                state = (
                    spike_record.float() * membrane_reset
                    + state * (spike_record ^ 1).float()
                )

            if threshold_low is not None and not neg_spikes:
                state = self.thresh_lower(state)  # Lower bound on the activation

            self.spikes_number += spikes[iCurrentTimeStep].abs().sum()

        self.state = state
        self.tw = len(spikes)
        return spikes

    @property
    def membrane_subtract(self):
        if self._membrane_subtract is not None:
            return self._membrane_subtract
        else:
            return self.threshold

    @membrane_subtract.setter
    def membrane_subtract(self, new_val):
        self._membrane_subtract = new_val
