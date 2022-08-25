import numpy as np
import torch.nn as nn
import torch
from typing import Optional, Union, List, Tuple
from sinabs.cnnutils import conv_output_size

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingMaxPooling2dLayer(nn.Module):
    """
    Torch implementation of SpikingMaxPooling.
    """

    def __init__(
        self,
        pool_size: ArrayLike,
        strides: Optional[ArrayLike] = None,
        padding: ArrayLike = (0, 0, 0, 0),
        # state_number: int = 16,
    ):
        super().__init__()
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
        # self.state_number = state_number

        # Blank parameter place holders
        self.spikes_number = None

    def forward(self, binary_input):
        # Determine no. of time steps from input
        time_steps = len(binary_input)

        # Calculate the cumulative sum spikes of each neuron
        sum_count = torch.cumsum(binary_input, 0)

        # max_sum is the pooled sum_count
        if self.pad is None:
            max_sum = self.pool(sum_count)
        else:
            max_sum = self.pool(self.pad(sum_count))

        # make sure a single spike, how much sum_count it brings, max_input_sum shows that
        input_sum = sum_count * (binary_input > 0).float()
        # pool all inputs at once
        if self.pad is None:
            max_input_sum = self.pool(input_sum)
        else:
            max_input_sum = self.pool(self.pad(input_sum))

        # max of 1 and 0 s from spike train
        if self.pad is None:
            original_max_input_sum = self.pool(binary_input)
        else:
            original_max_input_sum = self.pool(self.pad(binary_input))

        # Make sure the max sum is brought by the single spike from input_sum
        # (max_input_sum >= max_sum).float() is the gate to pass through spikes
        max_input_sum = (max_input_sum >= max_sum).float() * original_max_input_sum

        self.spikes_number = max_input_sum.abs().sum()
        self.tw = len(max_input_sum)
        return max_input_sum.float()  # Float is just to keep things compatible

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


class SumPool2d(torch.nn.LPPool2d):
    """
    Non-spiking sumpooling layer to be used in analogue Torch models. It is identical to torch.nn.LPPool2d with p=1.

    :param kernel_size: the size of the window
    :param stride: the stride of the window. Default value is kernel_size
    :param ceil_mode: when True, will use ceil instead of floor to compute the output shape
    """

    def __init__(self, kernel_size, stride=None, ceil_mode=False):
        super().__init__(1, kernel_size, stride, ceil_mode)
