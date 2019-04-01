##
# iaf_conv2d.py - Torch implementation of a spiking 2D convolutional layer
##

import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from typing import Optional, Union, List, Tuple, Dict
from operator import mul
from functools import reduce
from .layer import TorchLayer
from sinabs.cnnutils import conv_output_size, compute_padding

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingMaxPooling2dLayer(TorchLayer):
    def __init__(
        self,
        image_shape: ArrayLike,
        pool_size: ArrayLike,
        strides: Optional[ArrayLike] = None,
        padding: ArrayLike = (0, 0, 0, 0),
        layer_name: str = "pooling2d",
    ):
        """
        Torch implementation of SpikingMaxPooling
        """
        TorchLayer.__init__(
            self, input_shape=(None, *image_shape), layer_name=layer_name
        )
        self.padding = padding
        self.pool_size = pool_size
        if strides is None:
            strides = pool_size
        self.strides = strides
        if padding == (0, 0, 0, 0):
            self.pad = None
        else:
            self.pad = nn.ZeroPad2d(padding)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=strides)

        # Blank parameter place holders
        self.spikes_number = None
        self.state = None

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

        # Calculate the sum spikes of each neuron
        sum_count = torch.cumsum(binary_input, 0)
        # Initialize state as required
        # Create a vector to hold all output spikes
        if self.spikes_number is None or len(self.spikes_number) != len(binary_input):
            del self.spikes_number  # Free memory just to be sure
            self.spikes_number = sum_count.new_zeros(
                time_steps, *sum_count.shape[1:]
            ).int()

        self.spikes_number.zero_()
        spikes_number = self.spikes_number

        if self.state is None:
            self.state = sum_count.new_zeros(sum_count.shape[1:])

        state = self.state
        sum_count = torch.add(state, sum_count)

        # pool all inputs at once
        if self.pad is None:
            pool_sum_out = self.pool(sum_count)
        else:
            pool_sum_out = self.pool(self.pad(sum_count))

        # Get through spikes is input_spike
        input_spike = sum_count * (binary_input > 0).float()
        # input_spike = sum_count * binary_input  # tsrBinary cannot > 1

        # pool all inputs at once
        if self.pad is None:
            pool_out = self.pool(input_spike)
        else:
            pool_out = self.pool(self.pad(input_spike))

        # pool all inputs at once
        if self.pad is None:
            original_pool_out = self.pool(binary_input)
        else:
            original_pool_out = self.pool(self.pad(binary_input))

        # Make sure only the max count can pass the input tensor
        pool_out = (pool_out >= pool_sum_out).float() * original_pool_out
        # pool_out = (pool_out >= pool_sum_out) # tsrBinary cannot > 1

        self.state = sum_count[-1]
        self.spikes_number = pool_out
        return pool_out.float()  # Float is just to keep things compatible

    def summary(self):
        """
        Returns the summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Output_Shape": (tuple(self.output_shape)),
                "Input_Shape": (tuple(self.input_shape)),
                "Padding": tuple(self.padding),
                "Kernel": tuple(self.pool_size),
                "Pooling": tuple(self.pool_size),
                "Stride": tuple(self.strides),
                "Fanout_Prev": reduce(
                    mul, np.array(self.pool_size) / np.array(self.strides), 1
                ),
                "Neurons": 0,
                "Kernel_Params": 0,
                "Bias_Params": 0,
            }
        )
        return summary

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Returns the shape of output, given an input to this layer

        :param input_shape: (channels, height, width)
        :return: (channelsOut, height_out, width_out)
        """
        (channels, height, width) = input_shape

        height_out = conv_output_size(
            height + sum(self.padding[2:]), self.pool_size[0], self.strides[0]
        )
        width_out = conv_output_size(
            width + sum(self.padding[:2]), self.pool_size[1], self.strides[1]
        )
        return channels, height_out, width_out
