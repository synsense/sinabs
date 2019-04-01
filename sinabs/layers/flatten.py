import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from .layer import TorchLayer
from operator import mul
from functools import reduce

ArrayLike = Union[np.ndarray, List, Tuple]


class FlattenLayer(TorchLayer):
    """
    Equivalent to keras flatten
    """

    def __init__(self, input_shape, layer_name="flatten"):
        """
        Torch implementation of Flatten layer
        """
        TorchLayer.__init__(
            self, input_shape=input_shape, layer_name=layer_name
        )  # Init nn.Module
        self.layer_name = layer_name
        # TODO: should add ability to switch between channels first or channels last

    def forward(self, binary_input):
        nBatch = len(binary_input)
        # Temporary modify LQ, due to keras weights generation change
        # binary_input = binary_input.permute(0, 2, 3, 1)
        flatten_out = binary_input.contiguous().view(nBatch, -1)
        self.spikes_number = flatten_out
        return flatten_out

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        return (reduce(mul, self.input_shape),)

    def summary(self):
        """
        Returns a summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Input_Shape": (tuple(self.input_shape)),
                "Output_Shape": (tuple(self.output_shape)),
                "Fanout_Prev": 1,
                "Neurons": 0,
                "Kernel_Params": 0,
                "Bias_Params": 0,
            }
        )
        return summary


def from_flatten_keras_conf(
    layer_config: dict, input_shape: ArrayLike, spiking=False
) -> [(str, nn.Module)]:
    """
    Load Flatten layer from keras configuration
    :param layer_config: configuration dictionary
    :param input_shape: input data shape to determine output dimensions
    :param spiking: bool True if spiking layer needs to be loaded
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

    if spiking:
        # For a spiking layer, the next convolutional layer automatically re-formats the data structure
        return []
    else:
        # Determine output dims
        torch_layer = FlattenLayer(input_shape=input_shape, layer_name=layer_name)
        torch_layer.input_shape = input_shape

    return [(layer_name, torch_layer)]
