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

import torch
import numpy as np
import pandas as pd
from .layer import TorchLayer
from typing import Optional, Union, List, Tuple

ArrayLike = Union[np.ndarray, List, Tuple]


class QuantizeLayer(TorchLayer):
    """
    Equivalent to keras flatten
    """

    def __init__(self, layer_name: str = "quantize"):
        """
        Torch implementation of Quantizing the output spike count.

        :param layer_name: str Name of the layer
        """
        TorchLayer.__init__(self, input_shape=[])  # Init nn.Module
        self.layer_name = layer_name

    def forward(self, tsrInput):
        # Quantize to ints
        tsrOut: torch.Tensor = tsrInput.int().float()
        self.spikes_number = tsrOut.sum()
        return tsrOut

    def get_output_shape(self, input_shape: Tuple):
        """
        Shape of the input to this layer

        :param input_shape:
        :return: The output shape is identical to that of input
        """
        return input_shape

    def summary(self) -> pd.Series:
        """
        :return: A summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Neurons": 0,
                "Kernel_Params": 0,
                "Bias_Params": 0,
            }
        )
        return summary
