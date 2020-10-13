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

from typing import Union, List, Tuple
import numpy as np
from torch import nn

ArrayLike = Union[np.ndarray, List, Tuple]


class Cropping2dLayer(nn.Module):
    """
    Crop input image by
    """

    def __init__(
        self,
        cropping: ArrayLike = ((0, 0), (0, 0)),
    ):
        """
        Crop input to the the rectangle dimensions

        :param cropping: ((top, bottom), (left, right))
        """
        super().__init__()
        self.top_crop, self.bottom_crop = cropping[0]
        self.left_crop, self.right_crop = cropping[1]

    def forward(self, binary_input):
        _, self.channels_in, h, w = list(binary_input.shape)
        # Crop the data array
        crop_out = binary_input[
            :,
            :,
            self.top_crop: h - self.bottom_crop,
            self.left_crop: w - self.right_crop,
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
