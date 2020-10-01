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
from typing import Optional, Union, List, Tuple
from .functional import threshold_subtract

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

window = 1.0


class SpikingLayer(nn.Module):
    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = None,
        layer_name: str = "spiking",
        negative_spikes: bool = False,
        batch_size: Optional[int] = None,
        membrane_reset=None,
    ):
        """
        Pytorch implementation of a spiking neuron with learning enabled.
        NOTE: membrane_reset is ignored. Only does membrane subtract
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        :param input_shape: Input data shape
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: bool Upon spiking if the membrane potential is subtracted (True) as opposed to reset(False)
        :param layer_name: Name of this layer
        :param negative_spikes: Implement a linear transfer function through negative spiking
        """
        super().__init__()
        # Initialize neuron states
        self.threshold = threshold
        self.threshold_low = threshold_low
        self.negative_spikes = negative_spikes
        self._membrane_subtract = membrane_subtract

        # Blank parameter place holders
        self.register_buffer("state", torch.zeros(1))
        self.register_buffer("activations", torch.zeros(1))
        self.spikes_number = None
        self.batch_size = batch_size

        if membrane_reset is not None:
            raise NotImplementedError("Membrane reset is no longer supported.")

    #@property
    #def threshold_low(self):
    #    return self._threshold_low

    #@threshold_low.setter
    #def threshold_low(self, new_threshold_low):
    #    self._threshold_low = new_threshold_low
    #    #if new_threshold_low is None:
    #    #    try:
    #    #        del self.thresh_lower
    #    #    except AttributeError:
    #    #        pass
    #    #else:
    #    #    # Relu on the layer
    #    #    self.thresh_lower = nn.Threshold(new_threshold_low, new_threshold_low)

    @property
    def membrane_subtract(self):
        if self._membrane_subtract is not None:
            return self._membrane_subtract
        else:
            return self.threshold

    @membrane_subtract.setter
    def membrane_subtract(self, new_val):
        self._membrane_subtract = new_val

    def reset_states(self, shape=None, randomize=False):
        """
        Reset the state of all neurons in this layer
        """
        device = self.state.device
        if shape is None:
            shape = self.state.shape

        if randomize:
            self.state = torch.rand(shape, device=device)
            self.activations = torch.rand(shape, device=device)
        else:
            self.state = torch.zeros(shape, device=self.state.device)
            self.activations = torch.zeros(shape, device=self.activations.device)

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method needs to be overridden/defined by the child class
        Default implementation is pass through

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        return input_spikes

    def forward(self, binary_input: torch.Tensor):

        # Compute the synaptic current
        syn_out: torch.Tensor = self.synaptic_output(binary_input)

        # Reshape data to appropriate dimensions
        if self.batch_size:
            syn_out = syn_out.reshape((-1, self.batch_size, *syn_out.shape[1:]))
        # Ensure the neuron state are initialized
        try:
            assert self.state.shape == syn_out.shape[1:]
        except AssertionError:
            self.reset_states(shape=syn_out.shape[1:], randomize=False)

        # Determine no. of time steps from input
        time_steps = len(syn_out)

        # Local variables
        threshold = self.threshold
        threshold_low = self.threshold_low
        neg_spikes = self.negative_spikes

        state = self.state
        activations = self.activations
        spikes = []
        for iCurrentTimeStep in range(time_steps):
            # update neuron states
            state = syn_out[iCurrentTimeStep] + state - activations * threshold
            if threshold_low is not None and not neg_spikes:
                # This is equivalent to functional.threshold. non zero threshold is not supported for onnx
                state = torch.nn.functional.relu(state-threshold_low) + threshold_low
            # generate spikes
            if neg_spikes:
                activations = (
                    threshold_subtract(state.abs(), threshold, threshold * window)
                    * state.sign().int()
                )
            else:
                activations = threshold_subtract(state, threshold, threshold * window)
            spikes.append(activations)

        self.state = state
        self.tw = time_steps
        self.activations = activations
        all_spikes = torch.stack(spikes)
        self.spikes_number = all_spikes.abs().sum()

        if self.batch_size:
            all_spikes = all_spikes.reshape((-1, *all_spikes.shape[2:]))

        return all_spikes

    def get_output_shape(self, in_shape):
        """
        Returns the output shape for passthrough implementation

        :param in_shape:
        :return: out_shape
        """
        return in_shape

    def __deepcopy__(self, memo=None):
        other = SpikingLayer(
            threshold=self.threshold,
            threshold_low=self.threshold_low,
            membrane_subtract=self._membrane_subtract,
            layer_name="spiking",
            negative_spikes=self.negative_spikes,
            batch_size=self.batch_size,
            membrane_reset=None,
        )

        other.state = self.state.detach().clone()
        other.activations = self.activations.detach().clone()

        return other
