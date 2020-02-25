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

##
# iaf.py - Torch implementation of a integrate-and-fire layer
##

import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict
from .layer import Layer
from abc import abstractmethod
from .functional import threshold_subtract

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingLayer(Layer):
    def __init__(
        self,
        input_shape: ArrayLike,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = 1.0,
        membrane_reset: float = 0,
        layer_name: str = "spiking",
        negative_spikes: bool = False
    ):
        """
        Pytorch implementation of a spiking neuron with learning enabled.
        NOTE: membrane_reset is ignored. Only does membrane subtract
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        :param input_shape: Input data shape
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: Upon spiking if the membrane potential is subtracted as opposed to reset, what is its value
        :param membrane_reset: Only here for compatibility with other layers
        :param layer_name: Name of this layer
        :param negative_spikes: Implement a linear transfer function through negative spiking
        """
        super().__init__(input_shape=input_shape, layer_name=layer_name)
        # Initialize neuron states
        assert (membrane_subtract is not None)
        self.membrane_subtract = membrane_subtract
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.negative_spikes = negative_spikes

        # Blank parameter place holders
        self.register_buffer("state", torch.zeros(1))
        self.register_buffer("activations", torch.zeros(1))
        self.spikes_number = None

    @property
    def threshold_low(self):
        return self._threshold_low

    @threshold_low.setter
    def threshold_low(self, new_threshold_low):
        self._threshold_low = new_threshold_low
        if new_threshold_low is None:
            try:
                del self.thresh_lower
            except AttributeError:
                pass
        else:
            # Relu on the layer
            self.thresh_lower = nn.Threshold(new_threshold_low, new_threshold_low)

    def reset_states(self, shape=None):
        """
        Reset the state of all neurons in this layer
        """
        if shape is None:
            shape = self.state.shape
        else:
            self.state = torch.zeros(shape, device=self.state.device)
            self.activations = torch.zeros(shape, device=self.activations.device)

    @abstractmethod
    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method needs to be overridden/defined by the child class

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        pass

    def forward(self, binary_input: torch.Tensor):
        # Determine no. of time steps from input
        neg_spikes = self.negative_spikes

        # Compute the synaptic current
        syn_out: torch.Tensor = self.synaptic_output(binary_input)
        time_steps = len(syn_out)

        # Local variables
        threshold = self.threshold
        threshold_low = self.threshold_low

        # Initialize state as required
        if self.state.shape != syn_out.shape[1:]:
            self.reset_states(shape=syn_out.shape[1:])
        state = self.state
        activations = self.activations
        spikes = []
        for iCurrentTimeStep in range(time_steps):
            # update neuron states
            state = syn_out[iCurrentTimeStep] + state - activations * threshold
            if threshold_low is not None and not neg_spikes:
                state = self.thresh_lower(state)
            # generate spikes
            if neg_spikes:
                activations = threshold_subtract(state.abs(), threshold, threshold/2)*state.sign().int()
            else:
                activations = threshold_subtract(state, threshold, threshold / 2)
            spikes.append(activations)

        self.state = state
        self.tw = time_steps
        self.activations = activations
        all_spikes = torch.stack(spikes)
        self.spikes_number = all_spikes.abs().sum()
        return all_spikes
