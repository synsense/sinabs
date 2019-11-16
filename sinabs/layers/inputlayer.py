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

from .layer import TorchLayer
import pandas as pd
import numpy as np
from typing import Union, List, Tuple

ArrayLike = Union[np.ndarray, List, Tuple]


class InputLayer(TorchLayer):
    def __init__(self, input_shape: ArrayLike, layer_name="input"):
        """
        Torch implementation of SumPooling using the LPPool2d module

        :param image_shape: Input image dimensions
        """
        TorchLayer.__init__(self, input_shape=input_shape, layer_name=layer_name)

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

    def summary(self):
        """
        :return: A summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Input_Shape": tuple(self.input_shape),
                "Output_Shape": tuple(self.output_shape),
                "Fanout_Prev": 1,
                "Neurons": 0,
                "Kernel_Params": 0,
                "Bias_Params": 0,
            }
        )
        return summary


