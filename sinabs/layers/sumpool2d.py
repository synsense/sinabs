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
# sumpool2d.py -- Torch implementation of SumPooling2D layer (CNN architectures)
##

import torch.nn as nn
import numpy as np
import pandas as pd
from operator import mul
from functools import reduce
from collections import OrderedDict
from typing import Optional, Union, List, Tuple
from .layer import TorchLayer
from sinabs.cnnutils import conv_output_size, compute_padding

ArrayLike = Union[np.ndarray, List, Tuple]


class SumPooling2dLayer(TorchLayer):
    """
    Torch implementation of SumPooling2d for spiking neurons
    """

    def __init__(
        self,
        image_shape: ArrayLike,
        pool_size: ArrayLike,
        strides: Optional[ArrayLike] = None,
        padding: ArrayLike = (0, 0, 0, 0),
        layer_name: str = "pooling2d",
    ):
        """
        Torch implementation of SumPooling using the LPPool2d module

        :param image_shape:
        :param pool_size:
        :param strides:
        :param padding:
        :param layer_name:
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
        self.pool = nn.LPPool2d(1, kernel_size=pool_size, stride=strides)

    def forward(self, binary_input):
        if self.pad is None:
            pool_out = self.pool(binary_input)
        else:
            pool_out = self.pool(self.pad(binary_input))
        self.spikes_number = pool_out.sum().detach()
        return pool_out

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

    def get_output_shape(self, input_shape) -> Tuple:
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


def from_avgpool2d_keras_conf(
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
        spike_pool = SumPooling2dLayer(
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
        analogue_pool = nn.AvgPool2d(kernel_size=pool_size, stride=strides)
        layer_list.append((layer_name, analogue_pool))

    if len(layer_list) > 1:
        return [(layer_name, nn.Sequential(OrderedDict(layer_list)))]
    else:
        return layer_list


def from_sumpool2d_keras_conf(
    layer_config, input_shape: Tuple, spiking: bool = False
) -> [(list, nn.Module)]:
    """
    Creates AveragePooling layer from keras configuration file
    This is actually a proxy to SumPooling2dLayer in case of spiking version

    :param layer_config:
    :param input_shape:
    :param spiking: The spiking and non spiking layers act the same for sum pooling, so this option is ignored
    :return: [(layer_name, nn.Module)] Returns a list of layers and their names
    """
    return from_avgpool2d_keras_conf(layer_config, input_shape, spiking=True)
