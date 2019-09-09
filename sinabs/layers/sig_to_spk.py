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
        return torch.stack([input_sig]*self.tw, dim=0).float()

    def forward(self, binary_input: torch.Tensor) -> torch.Tensor:
        # Compute the synaptic current
        syn_out: torch.Tensor = self.synaptic_output(binary_input)

        # Determine no. of time steps from input
        time_steps = self.tw

        # Local variables
        membrane_subtract = self.membrane_subtract
        threshold = self.threshold
        threshold_low = self.threshold_low
        membrane_reset = self.membrane_reset

        # Initialize state as required
        # Create a vector to hold all output spikes
        if self.spikes_number is None or len(self.spikes_number) != time_steps:
            del self.spikes_number  # Free memory just to be sure
            self.spikes_number = syn_out.new_zeros(time_steps, *syn_out.shape[1:]).int()

        self.spikes_number.zero_()
        spikes_number = self.spikes_number

        if self.state is None:
            self.state = syn_out.new_zeros(syn_out.shape[1:])
        elif self.state.device != syn_out.device:
            # print(f"Device type state: {self.state.device}, syn_out: {syn_out.device} ")
            self.state = self.state.to(syn_out.device)

        state = self.state

        # Loop over time steps
        for iCurrentTimeStep in range(time_steps):
            state = state + syn_out[iCurrentTimeStep]

            # - Reset or subtract from membrane state after spikes
            if membrane_subtract is not None:
                # Calculate number of spikes to be generated
                spikes_number[iCurrentTimeStep] = (state >= threshold).int() + (
                    state - threshold > 0
                ).int() * ((state - threshold) / membrane_subtract).int()
                # - Subtract from states
                state = state - (
                    membrane_subtract * spikes_number[iCurrentTimeStep].float()
                )
            else:
                # - Check threshold crossings for spikes
                spike_record = state >= threshold
                # - Add to spike counter
                spikes_number[iCurrentTimeStep] = spike_record
                # - Reset neuron states
                state = (
                    spike_record.float() * membrane_reset
                    + state * (spike_record ^ 1).float()
                )

            if threshold_low is not None:
                state = self.thresh_lower(state)  # Lower bound on the activation

        self.state = state
        self.spikes_number = spikes_number
        return spikes_number.float()  # Float to keep things compatible

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
