"""
This is a base class for all Torch layers (especially for CNN layers)
"""
from torch import nn
import numpy as np
from typing import Tuple, Optional, Union, List
from abc import ABC, abstractmethod

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class TorchLayer(nn.Module, ABC):
    def __init__(self, input_shape: ArrayLike, layer_name: str = ""):
        """
        Base class for all torch layers

        :param input_shape: Array like with channels first notation
        :param layer_name: str Name of the layer
        """
        nn.Module.__init__(self)
        ABC.__init__(self)
        # Instantiate all input variables
        self.input_shape = input_shape
        self._output_shape: Optional[ArrayLike] = None
        self.layer_name = layer_name

    @abstractmethod
    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Abstract method, needs to be defined
        This method computes the output dimensions given a certain input

        :param input_shape: Tuple, shape of input
        :return: output_shape Tuple
        """
        pass

    def evolve(self, tsInput, tDuration: float, time_steps: int, bVerbose: bool):
        """
        Convenience function for eventual merger/compatibility with NetworksPython

        :param tsInput:
        :param tDuration:
        :param time_steps:
        :param bVerbose:
        :return:
        """
        return self.forward(tsInput)

    @property
    def output_shape(self) -> Tuple:
        """
        :return: Return the output dimensions of the layer
        """
        if self._output_shape is None:
            # Compute output dimensions
            self._output_shape = self.get_output_shape(self.input_shape)
        return self._output_shape
