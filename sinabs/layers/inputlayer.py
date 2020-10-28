from torch import nn
import numpy as np
from typing import Union, List, Tuple

ArrayLike = Union[np.ndarray, List, Tuple]


class InputLayer(nn.Module):
    def __init__(self, input_shape: ArrayLike, layer_name="input"):
        """
        Place holder layer, used typically to acquire some statistics on the input

        :param image_shape: Input image dimensions
        """
        super().__init__()
        self.input_shape = input_shape

    def forward(self, binary_input):
        """
        Passthrough layer

        :param binary_input:
        :return: binary_input
        """
        self.spikes_number = binary_input.abs().sum()
        self.tw = len(binary_input)
        return binary_input

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Retuns the output dimensions

        :param input_shape: (channels, height, width)
        :return: (channels, height, width)
        """
        return self.input_shape
