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

import warnings
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from .layer import Layer

ArrayLike = Union[np.ndarray, List, Tuple]


class ZeroPad2dLayer(Layer):
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
        super().__init__(input_shape=(None, *image_shape), layer_name=layer_name)
        warnings.warn(
            "ZeroPad2dLayer deprecated. Use nn.ZeroPad2d instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding)

    def forward(self, tsrInput):
        output = self.pad(tsrInput)
        self.spikes_number = output.abs().sum()
        self.tw = len(output)
        return output

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        channels, height, width = input_shape
        return channels, height + sum(self.padding[2:]), width + sum(self.padding[:2])

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
