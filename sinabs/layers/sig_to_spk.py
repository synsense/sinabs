#  Copyright (c) 2019-2019     aiCTX AG (Sadique Sheik, Massimo Bortone).
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
from .layer import Layer
from typing import Tuple, Optional
import torch


class Sig2SpikeLayer(Layer):
    """
    Layer to convert analog Signals to Spikes
    """

    def __init__(
        self,
        channels_in,
        tw: int = 1,
        layer_name: str = "sig2spk",
    ):
        """
        Layer converts analog signals to spikes

        :param channels_in: number of channels in the analog signal
        :param tw: int number of time steps for each sample of the signal (up sampling)
        :param layer_name: string layer name

        """
        self.tw = tw
        super().__init__(
            input_shape=(channels_in, None),
            layer_name=layer_name
        )


    def get_output_shape(self, input_shape: Tuple):
        channels, time_steps = input_shape
        return (self.tw*time_steps, channels)


    def forward(self, signal):
        channels, time_steps = signal.shape
        random_tensor = torch.rand(self.tw*time_steps, channels).to(signal.device)
        if self.tw != 1:
            signal = signal.view(-1,1).repeat(1, self.tw).view(channels, -1)
        signal = signal.transpose(1,0)
        spk_sig= (random_tensor < signal).float()
        self.spikes_number = spk_sig.abs().sum()
        return spk_sig

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
                "Neurons": self.output_shape[1],
            }
        )
        return summary
