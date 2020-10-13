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

from typing import Tuple
import torch


class Sig2SpikeLayer(torch.nn.Module):
    """
    Layer to convert analog Signals to Spikes
    """

    def __init__(
        self,
        channels_in,
        tw: int = 1,
        norm_level: float = 1,
        layer_name: str = "sig2spk",
        spk_out: bool = True,
    ):
        """
        Layer converts analog signals to spikes

        :param channels_in: number of channels in the analog signal
        :param tw: int number of time steps for each sample of the signal (up sampling)
        :param layer_name: string layer name

        """
        super().__init__()
        self.tw = tw
        self.norm_level = norm_level
        self.spk_out = spk_out

    def get_output_shape(self, input_shape: Tuple):
        channels, time_steps = input_shape
        return (self.tw * time_steps, channels)

    def forward(self, signal):
        """
        Convert a signal to the corresponding spikes

        :param signal: [Channel, Sample(t)]
        :return:
        """
        channels, time_steps = signal.shape
        if self.tw != 1:
            signal = signal.view(-1, 1).repeat(1, self.tw).view(channels, -1)
        signal = signal.transpose(1, 0)
        if self.spk_out:
            random_tensor = (
                torch.rand(self.tw * time_steps, channels).to(signal.device)
                * self.norm_level
            )
            spk_sig = (random_tensor < signal).float()
        else:
            # If there is no conversion to spikes
            # just replicate the signal as current injection
            spk_sig = signal

        self.spikes_number = spk_sig.abs().sum()
        return spk_sig