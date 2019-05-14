#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik).
#
#  This file is part of sinabs
#
#  sinabs is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  sinabs is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with sinabs.  If not, see <https://www.gnu.org/licenses/>.

##
# iaf_conv3d.py - Torch implementation of a spiking 3D convolutional layer
##
import torch
import torch.nn as nn
import numpy as np
from .iaf import SpikingLayer
from typing import Optional, Union, List, Tuple, Dict

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingConv3dLayer(SpikingLayer):
    def __init__(
        self,
        channels_in: int,
        image_shape: ArrayLike,
        channels_out: int,
        kernel_shape: ArrayLike,
        strides: ArrayLike = (1, 1, 1),
        padding: ArrayLike = (0, 0, 0, 0, 0, 0),
        bias: bool = True,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = 1.0,
        membrane_reset: float = 0,
        layer_name: str = "conv3d",
    ):
        """
        Spiking 3D convolutional layer

        :param channels_in: Number of input channels
        :param image_shape: [Height, Width]
        :param channels_out: Number of output channels
        :param kernel_shape: Size of the kernel  (tuple)
        :param strides: Strides in each direction (tuple of size 3)
        :param padding: Padding in each of the 6 directions (left, right, top, bottom, front, back)
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: Upon spiking if the membrane potential is subtracted as opposed to reset, what is its value
        :param membrane_reset: What is the reset membrane potential of the neuron
        :param layer_name: Name of this layer

        NOTE: SUBTRACT superseeds Reset value
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
        self.conv = nn.Conv3d(
            channels_in,
            channels_out,
            kernel_size=kernel_shape,
            stride=strides,
            bias=bias,
        )

        # Layer convolutional properties
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = kernel_shape
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
        if self.pad is None:
            syn_out = self.conv(input_spikes)
        else:
            syn_out = self.conv(self.pad(input_spikes))
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
        (channels, height, width) = input_shape

        height_out = conv_output_size(
            height + sum(self.padding[2:]), self.kernel_shape[0], self.strides[0]
        )
        width_out = conv_output_size(
            width + sum(self.padding[:2]), self.kernel_shape[1], self.strides[1]
        )
        return self.channels_out, height_out, width_out
