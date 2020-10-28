import torch.nn as nn
import numpy as np
import pandas as pd
from operator import mul
from functools import reduce
from typing import Optional, Union, List, Tuple
from .layer import Layer
from sinabs.cnnutils import conv_output_size

ArrayLike = Union[np.ndarray, List, Tuple]


class SumPooling2dLayer(Layer):
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
        Layer.__init__(
            self, input_shape=(None, *image_shape), layer_name=layer_name
        )
        self.padding = padding
        self.pool_size = pool_size
        if (strides is None) or (None in strides):
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
        self.spikes_number = pool_out.abs().sum()
        self.tw = len(pool_out)
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


