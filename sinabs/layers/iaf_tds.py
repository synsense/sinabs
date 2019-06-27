#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Massimo Bortone).
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

from .iaf import SpikingLayer
from functools import reduce
from operator import mul
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from typing import Optional, Tuple
from ..cnnutils import conv_output_size


class SpikingTDSLayer(SpikingLayer):
    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            delay: int,
            bias: bool = True,
            threshold: float = 1.0,
            threshold_low: Optional[float] = -1.0,
            membrane_subtract: Optional[float] = 1.0,
            membrane_reset: float = 0,
            layer_name: str = "conv1d",
    ):
        """
        Temporal Delay Spiking layer. This layer performs wave net like streaming computation,
        where the neuron convolves input at time `t` and time `t-delay` to produce output at time `t`

        :param channels_in: Number of input channels
        :param channels_out: Number of output channels
        :param delay: int Number of simulations time steps in the past to look for as input,
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lower bound for membrane potential
        :param membrane_subtract: Upon spiking if the membrane potential is subtracted as opposed to reset, what is its value
        :param membrane_reset: What is the reset membrane potential of the neuron
        :param layer_name: Name of this layer

        NOTE: SUBTRACT superseeds Reset value
        """
        SpikingLayer.__init__(
            self,
            input_shape=(channels_in, 1),
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            layer_name=layer_name,
        )
        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=2,
            padding=delay,
            dilation=delay,
            stride=1,
            bias=bias,
        )
        # Initialize buffer
        self.register_buffer("delay_buffer", torch.zeros((1, channels_in, delay)))

        # Layer convolutional properties
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = 2
        self.delay = delay
        self.bias = bias

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method convolves the input spikes to compute the synaptic input currents to the neuron states

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        # Convolve all inputs at once
        input_spikes = torch.transpose(input_spikes, 0, 2)
        input_spikes = torch.cat((self.delay_buffer, input_spikes), 2)
        syn_out = self.conv(input_spikes)
        self.delay_buffer = input_spikes[:, : , -self.delay:]
        syn_out = syn_out[:, :, self.delay:-self.delay]
        syn_out = torch.transpose(syn_out, 0, 2)
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
                "Output_Shape": self.output_shape,
                "Input_Shape": self.input_shape,
                "Kernel": self.kernel_shape,
                "Padding": tuple(self.padding),
                "Stride": self.strides,
                "Fanout_Prev": self.kernel_shape
                               / np.array(self.strides)
                               * self.channels_out,
                "Neurons": reduce(mul, list(self.output_shape), 1),
                "Kernel_Params": self.channels_in*self.channels_out*self.kernel_shape,
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
        (channels, length) = input_shape

        length_out = conv_output_size(
            length + sum(self.padding[1]),
            (self.dilation * (self.kernel_shape - 1) + 1),
            self.strides,
            )
        return self.channels_out, length_out


