##
# iaf_conv2d.py - Torch implementation of a spiking 2D convolutional layer
##

import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict
from operator import mul
from functools import reduce
from .layer import TorchLayer
from .quantize import QuantizeLayer
from collections import OrderedDict
from sinabs.cnnutils import conv_output_size, compute_padding
from abc import ABC, abstractmethod

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingLayer(TorchLayer):
    def __init__(
        self,
        channels_in: int,
        image_shape: ArrayLike,
        threshold: float = 8,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        membrane_reset: float = 0,
        layer_name: str = "conv2d",
    ):
        """
        Pytorch implementation of a spiking neuron which convolve 2D inputs, with multiple channels

        :param channels_in: Number of input channels
        :param image_shape: [Height, Width]
        :param channels_out: Number of output channels
        :param kernel_shape: Size of the kernel  (tuple)
        :param strides: Strides in each direction (tuple of size 2)
        :param padding: Padding in each of the 4 directions (left, right, top, bottom)
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: Upon spiking if the membrane potential is subtracted as opposed to reset, what is its value
        :param membrane_reset: What is the reset membrane potential of the neuron
        :param layer_name: Name of this layer

        NOTE: SUBTRACT superseeds Reset value
        """
        TorchLayer.__init__(
            self, input_shape=(channels_in, *image_shape), layer_name=layer_name
        )
        if padding != (0, 0, 0, 0):
            self.pad = nn.ZeroPad2d(padding)
        else:
            self.pad = None
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_shape,
            stride=strides,
            bias=bias,
        )
        # Initialize neuron states
        self.membrane_subtract = membrane_subtract
        self.membrane_reset = membrane_reset
        self.threshold = threshold
        self.threshold_low = threshold_low

        # Layer convolutional properties
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.strides = strides
        self.bias = bias

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

    def forward(self, binary_input):
        # Determine no. of time steps from input
        time_steps = len(binary_input)

        # Convolve all inputs at once
        if self.pad is None:
            tsrConvOut = self.conv(binary_input)
        else:
            tsrConvOut = self.conv(self.pad(binary_input))

        # Local variables
        membrane_subtract = self.membrane_subtract
        threshold = self.threshold
        threshold_low = self.threshold_low
        membrane_reset = self.membrane_reset

        # Initialize state as required
        # Create a vector to hold all output spikes
        if self.spikes_number is None or len(self.spikes_number) != len(binary_input):
            del self.spikes_number  # Free memory just to be sure
            self.spikes_number = tsrConvOut.new_zeros(
                time_steps, *tsrConvOut.shape[1:]
            ).int()

        self.spikes_number.zero_()
        spikes_number = self.spikes_number

        if self.state is None:
            self.state = tsrConvOut.new_zeros(tsrConvOut.shape[1:])

        state = self.state

        # Loop over time steps
        for iCurrentTimeStep in range(time_steps):
            state = state + tsrConvOut[iCurrentTimeStep]

            # - Reset or subtract from membrane state after spikes
            if membrane_subtract is not None:
                # Calculate number of spikes to be generated
                spikes_number[iCurrentTimeStep] = (state >= threshold).int() + (
                    state - threshold > 0
                ).int() * ((state - threshold) / membrane_subtract).int()
                # - Subtract from states
                state = state - (
                    membrane_subtract * spikes_number[iCurrentTimeStep].float()
                )
            else:
                # - Check threshold crossings for spikes
                vbRecSpikeRaster = state >= threshold
                # - Add to spike counter
                spikes_number[iCurrentTimeStep] = vbRecSpikeRaster
                # - Reset neuron states
                state = (
                    vbRecSpikeRaster.float() * membrane_reset
                    + state * (vbRecSpikeRaster ^ 1).float()
                )

            if threshold_low is not None:
                state = self.thresh_lower(state)  # Lower bound on the activation

        self.state = state
        self.spikes_number = spikes_number
        return spikes_number.float()  # Float is just to keep things compatible



