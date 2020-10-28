import numpy as np
import pandas as pd
from typing import Union, List, Tuple
from .layer import Layer
from operator import mul
from functools import reduce
import warnings

ArrayLike = Union[np.ndarray, List, Tuple]


class FlattenLayer(Layer):
    """
    Equivalent to keras flatten
    """

    def __init__(self, input_shape, layer_name="flatten"):
        """
        Torch implementation of Flatten layer
        """
        super().__init__(input_shape=input_shape, layer_name=layer_name)  # Init nn.Module
        warnings.warn(
            "sinabs.layers.FlattenLayer deprecated. Use torch.nn.Flatten instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.layer_name = layer_name
        # TODO: should add ability to switch between channels first or channels last

    def forward(self, binary_input):
        nBatch = len(binary_input)
        # Temporary modify LQ, due to keras weights generation change
        # binary_input = binary_input.permute(0, 2, 3, 1)
        flatten_out = binary_input.contiguous().view(nBatch, -1)
        self.spikes_number = flatten_out.abs().sum()
        self.tw = len(flatten_out)
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
