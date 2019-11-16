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

import numpy as np
import pandas as pd
from typing import Union, List, Tuple
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


