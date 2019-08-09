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
from .iaf import SpikingLayer
from typing import Tuple
import torch


class Sig2SpikeLayer(SpikingLayer):
    """
    Layer to convert analog Signals to Spikes
    """

    def __init__(
        self,
        sig_shape,
        tw: int = 100,
        threshold: float = 1.0,
        layer_name: str = "sig2spk",
    ):
        """
        Layer converts analog signals to spikes

        :param sig_shape: shape of the analog signal (channels, length)
        :param tw: int Time window length
        :param threshold: Spiking threshold of the neuron
        :param layer_name: string layer name
        """
        SpikingLayer.__init__(
            self,
            input_shape=(None, *sig_shape),
            threshold=threshold,
            threshold_low=-threshold,
            membrane_subtract=threshold,
            membrane_reset=0.0,
            layer_name=layer_name
        )
        self.tw = tw

    def synaptic_output(self, input_sig: torch.Tensor) -> torch.Tensor:
        return input_sig

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
                "Neurons": self.output_shape[1],
            }
        )
        return summary
