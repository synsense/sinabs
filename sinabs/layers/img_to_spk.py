#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik).
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
from .layer import TorchLayer
from typing import Tuple
import torch


class Img2SpikeLayer(TorchLayer):
    """
    Layer to convert Images to Spikes
    """

    def __init__(
        self,
        image_shape,
        tw: int = 100,
        max_rate: float = 1000,
        layer_name: str = "img2spk",
    ):
        """
        Layer converts images to spikes

        :param image_shape: tuple image shape
        :param tw: int Time window length
        :param max_rate: maximum firing rate of neurons
        :param layer_name: string layer name
        """
        TorchLayer.__init__(
            self, input_shape=(None, *image_shape), layer_name=layer_name
        )
        self.tw = tw
        self.max_rate = max_rate

    def forward(self, img_input):
        spk_img = (
            (torch.rand(self.tw, *tuple(img_input.shape))) < (img_input/255.0)*(self.max_rate/1000)
        ).float()
        return spk_img

    def get_output_shape(self, input_shape: Tuple):
        return (self.tw, *input_shape)

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
                "Neurons": 0,
            }
        )
        return summary
