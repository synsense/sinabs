#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Qian Liu).
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
        # state_number: int = 16,
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
        # self.state_number = state_number

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
        self.reset_states()
        time_steps = len(binary_input)

        # Calculate the sum spikes of each neuron
        sum_count = torch.cumsum(binary_input, 0)
        # Initialize state as required
        # Create a vector to hold all output spikes
        if self.spikes_number is None:
            del self.spikes_number  # Free memory just to be sure
            self.spikes_number = torch.tensor(())

        self.spikes_number.zero_()
        spikes_number = self.spikes_number

        if self.state is None:
            self.state = sum_count.new_zeros(sum_count.shape[1:])

        state = self.state
        sum_count = torch.add(state, sum_count)

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
        max_input_sum = (max_input_sum >= max_sum).float() * original_max_input_sum

        self.state = sum_count[-1]
        self.spikes_number = max_input_sum.abs().sum()
        self.tw = len(max_input_sum)
        return max_input_sum.float()  # Float is just to keep things compatible

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


def from_maxpool2d_keras_conf(
    layer_config, input_shape: Tuple, spiking: bool = False
) -> [(list, nn.Module)]:
    """
    Crete a Average pooling layer

    :param layer_config:
    :param input_shape:
    :param spiking:
    :return:
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    layer_list = []
    pool_size = layer_config["config"]["pool_size"]
    strides = layer_config["config"]["strides"]
    pad_mod = layer_config["config"]["padding"]

    if pad_mod == "valid":
        padding = (0, 0, 0, 0)
    else:
        # Compute padding
        padding = compute_padding(pool_size, input_shape, pad_mod)

    if spiking:
        spike_pool = SpikingMaxPooling2dLayer(
            image_shape=input_shape[1:],
            pool_size=pool_size,
            padding=padding,
            strides=strides,
            layer_name=layer_name,
        )
        spike_pool.input_shape = input_shape
        layer_list.append((layer_name, spike_pool))
    else:
        # Create a padding layer
        if padding != (0, 0, 0, 0):
            torch_layerPad = nn.ZeroPad2d(padding)
            layer_list.append((layer_name + "_padding", torch_layerPad))
        # Pooling layer initialization
        analogue_pool = nn.MaxPool2d(kernel_size=pool_size, stride=strides)
        layer_list.append((layer_name, analogue_pool))

    if len(layer_list) > 1:
        return [(layer_name, nn.Sequential(OrderedDict(layer_list)))]
    else:
        return layer_list
