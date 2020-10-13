"""
This is a base class for all Torch layers (especially for CNN layers)
"""
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

from torch import nn
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union, List
from abc import ABC, abstractmethod
from operator import mul
from functools import reduce
import warnings

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class Layer(nn.Module, ABC):
    def __init__(self, input_shape: ArrayLike, layer_name: str = ""):
        """
        Base class for all torch layers

        :param input_shape: Array like with channels first notation
        :param layer_name: str Name of the layer
        """
        nn.Module.__init__(self)
        ABC.__init__(self)
        warnings.warn(f"Layer {self.__class__.__name__} is deprecated.")
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
        Convenience function for eventual merger/compatibility with Rockpool

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
                "Neurons": reduce(mul, list(self.output_shape), 1),
            }
        )
        return summary


class TorchLayer(Layer):
    def __init__(self, input_shape: ArrayLike, layer_name: str = ""):
        warnings.warn(
            "TorchLayer is deprecated and renamed to Layer",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(input_shape=input_shape, layer_name=layer_name)
