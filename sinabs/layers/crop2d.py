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

import pandas as pd
from typing import Union, List, Tuple
from .layer import TorchLayer
import numpy as np

ArrayLike = Union[np.ndarray, List, Tuple]


class Cropping2dLayer(TorchLayer):
    """
    Torch implementation of SumPooling2d for spiking neurons
    """

    def __init__(
        self,
        image_shape: ArrayLike,
        cropping: ArrayLike = ((0, 0), (0, 0)),
        layer_name="crop2d",
    ):
        """
        Torch implementation of SumPooling using the LPPool2d module

        :param image_shape: Input image dimensions
        :param layer_name: str Layer name
        """
        TorchLayer.__init__(
            self, input_shape=(None, *image_shape), layer_name=layer_name
        )
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]

    def forward(self, binary_input):
        _, self.channels_in, h, w = list(binary_input.shape)
        # Crop the data array
        crop_out = binary_input[
            :,
            :,
            self.top_crop : h - self.bottom_crop,
            self.left_crop : w - self.right_crop,
        ]
        self.out_shape = crop_out.shape[1:]
        self.spikes_number = crop_out.abs().sum()
        self.tw = len(crop_out)
        return crop_out

    def get_output_shape(self, input_shape: Tuple) -> Tuple:
        """
        Retuns the output dimensions

        :param input_shape: (channels, height, width)
        :return: (channels, height, width)
        """
        channels, height, width = input_shape
        return (
            channels,
            height - self.top_crop - self.bottom_crop,
            width - self.left_crop - self.right_crop,
        )

    def summary(self):
        """
        Returns a summary of this layer as a pandas Series
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Input_Shape": tuple(self.input_shape),
                "Output_Shape": tuple(self.output_shape),
                "Cropping": (
                    self.top_crop,
                    self.bottom_crop,
                    self.left_crop,
                    self.right_crop,
                ),
                "Fanout_Prev": 1,
                "Neurons": 0,
                "Kernel_Params": 0,
                "Bias_Params": 0,
            }
        )
        return summary


