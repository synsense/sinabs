import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from .layer import TorchLayer

ArrayLike = Union[np.ndarray, List, Tuple]


class ZeroPad2dLayer(TorchLayer):
    """
    Zero padding 2D layer
    """
    def __init__(self, image_shape, padding: ArrayLike, layer_name: str = "zeropad2d"):
        """
        Zero Padding Layer

        :param image_shape: Shape of the input image (height, width)
        :param padding: No. of pixels to pad on each side (left, right, top, bottom)
        :param layer_name: Name of the layer
        """
        TorchLayer.__init__(
            self, input_shape=(None, *image_shape), layer_name=layer_name
        )
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)

    def forward(self, tsrInput):
        output = self.pad(tsrInput)
        self.spikes_number = output
        return output

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        channels, height, width = input_shape
        return (channels, height + sum(self.padding[2:]), width + sum(self.padding[:2]))

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
                "Fanout_Prev": 1,
                "Neurons": 0,
                "Kernel_Params": 0,
                "Bias_Params": 0,
            }
        )
        return summary


def from_zeropad2d_keras_conf(
    layer_config, input_shape: Tuple, spiking: bool = False
) -> [(str, nn.Module)]:
    """
    Load ZeroPadding layer from Json configuration

    :param layer_config: configuration dictionary
    :param input_shape: input data shape to determine output dimensions
    :param spiking: bool True if spiking layer needs to be loaded (This parameter is immaterial for this layer)
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
    # Determing output dims
    tb, lr = layer_config["config"]["padding"]

    torch_layer = ZeroPad2dLayer(
        image_shape=input_shape[1:], padding=(*lr, *tb), layer_name=layer_name
    )
    torch_layer.input_shape = input_shape
    return [(layer_name, torch_layer)]
