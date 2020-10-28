import torch
import warnings
import torch.nn as nn
import numpy as np
import pandas as pd
from functools import reduce
from operator import mul
from .iaf import SpikingLayer
from typing import Optional, Union, List, Tuple
from sinabs.cnnutils import conv_output_size
from torch.nn import functional

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingConv3dLayer(SpikingLayer):
    def __init__(
        self,
        channels_in: int,
        image_shape: ArrayLike,
        channels_out: int,
        kernel_shape: ArrayLike,
        dilation: ArrayLike = (1, 1, 1),
        strides: ArrayLike = (1, 1, 1),
        padding: ArrayLike = (0, 0, 0, 0, 0, 0),
        bias: bool = True,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset: Optional[float] = None,
        layer_name: str = "conv3d",
    ):
        """
        Spiking 3D convolutional layer

        :param channels_in: Number of input channels
        :param image_shape: [Height, Width]
        :param channels_out: Number of output channels
        :param kernel_shape: Size of the kernel  (tuple)
        :param dilation: kernel dilaiton (vertical_dilation, horizontal_dilation)
        :param strides: Strides in each direction (tuple of size 3)
        :param padding: Padding in each of the 6 directions (left, right, top, bottom, front, back)
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: Upon spiking, if the membrane potential is subtracted as opposed to reset, \
        what is the subtracted value? Defaults to threshold.
        :param membrane_reset: What is the reset membrane potential of the neuron. \
        If not None, the membrane potential is reset instead of subtracted on spiking.
        :param layer_name: Name of this layer
        """
        SpikingLayer.__init__(
            self,
            input_shape=(channels_in, *image_shape),
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            layer_name=layer_name,
        )
        warnings.warn(
            "SpikingConv3dLayer deprecated. Use nn.Conv3d + SpikingLayer instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.conv = nn.Conv3d(
            channels_in,
            channels_out,
            kernel_size=kernel_shape,
            dilation=dilation,
            stride=strides,
            bias=bias,
        )

        # Layer convolutional properties
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = kernel_shape
        self.dilation = dilation
        self.padding = padding
        self.strides = strides
        self.bias = bias

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method convolves the input spikes to compute the synaptic input currents to the neuron states

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        # Convolve all inputs at once
        if self.padding == (0, 0, 0, 0, 0, 0):
            syn_out = self.conv(input_spikes)
        else:
            # Zeropadded input
            syn_out = self.conv(functional.pad(input_spikes, self.padding, mode='constant', value=0))
        return syn_out

    def summary(self) -> pd.Series:
        """
        Returns a summary of the current layer

        :return: pandas Series object
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Output_Shape": tuple(self.output_shape),
                "Input_Shape": tuple(self.input_shape),
                "Kernel": tuple(self.kernel_shape),
                "Padding": tuple(self.padding),
                "Stride": tuple(self.strides),
                "Fanout_Prev": reduce(
                    mul, np.array(self.kernel_shape) / np.array(self.strides), 1
                )
                * self.channels_out,
                "Neurons": reduce(mul, list(self.output_shape), 1),
                "Kernel_Params": self.channels_in
                * self.channels_out
                * reduce(mul, self.kernel_shape, 1),
                "Bias_Params": self.bias * self.channels_out,
            }
        )
        return summary

    def get_output_shape(self, input_shape) -> Tuple:
        """
        Returns the shape of output, given an input to this layer

        :param input_shape: (channels, height, width)
        :return: (channelsOut, height_out, width_out)
        """
        (channels, depth, height, width) = input_shape

        depth_out = conv_output_size(
            height + sum(self.padding[4:]),
            (self.dilation[0] * (self.kernel_shape[0] - 1) + 1),
            self.strides[0],
        )
        height_out = conv_output_size(
            width + sum(self.padding[2:4]),
            (self.dilation[1] * (self.kernel_shape[1] - 1) + 1),
            self.strides[1],
        )
        width_out = conv_output_size(
            width + sum(self.padding[:2]),
            (self.dilation[2] * (self.kernel_shape[2] - 1) + 1),
            self.strides[2],
        )
        return self.channels_out, depth_out, height_out, width_out
