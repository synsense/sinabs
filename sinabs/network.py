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

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple
from .utils import (
    get_network_activations,
    get_activations,
)
from .layers import SpikingLayer

ArrayLike = Union[np.ndarray, List, Tuple]


class Network(torch.nn.Module):
    """
    Class of a spiking neural network

    Attributes:
        spiking_model: torch.nn.Module, a spiking neural network model
        analog_model: torch.nn.Module, an artifical neural network model
        graph: pandas DataFrame
        input_shape: Tuple, size of input
        quantize_activation: bool, if true, the analog model will be initialized
                           with a quantization layer after each activation

    """

    def __init__(
        self,
        analog_model: Optional = None,
        spiking_model: Optional = None,
        input_shape: Optional[ArrayLike] = None,
        quantize_activation: bool = False,
        nbit_quantize: Optional[int] = None,
    ):
        super().__init__()
        self.spiking_model: nn.Module = spiking_model
        self.analog_model: nn.Module = analog_model
        self.graph: pd.DataFrame = None
        self.input_shape = input_shape
        self.quantize_activation = quantize_activation

        if input_shape is not None and spiking_model is not None:
            self._compute_shapes(input_shape)

    @property
    def layers(self):
        return list(self.spiking_model.named_children())

    def _compute_shapes(self, input_shape, batch_size=1):
        def hook(module, inp, out):
            module.out_shape = out.shape[1:]

        hook_list = []
        for layer in self.spiking_model.modules():
            this_hook = layer.register_forward_hook(hook)
            hook_list.append(this_hook)

        # do a forward pass
        device = next(self.parameters()).device
        dummy_input = torch.zeros(
            [batch_size] + list(input_shape),
            requires_grad=False
        ).to(device)
        self(dummy_input)

        [this_hook.remove() for this_hook in hook_list]

    def forward(self, tsrInput) -> torch.Tensor:
        """
        Forward pass for this model
        """
        with torch.no_grad():
            return self.spiking_model(tsrInput)

    def compare_activations(
        self,
        data,
        name_list: Optional[ArrayLike] = None,
        compute_rate: bool = False,
        verbose: bool = False,
    ) -> ([np.ndarray], [np.ndarray]):
        """
        Compare activations of the analog model and the SNN for a given data sample

        :param data: Data to process
        :param name_list: list of all layer names (str) whose activations need to be compared
        :param compute_rate: True if you want to compute firing rate. By default spike count is returned
        :param verbose: bool print debugging logs to the terminal
        """
        if name_list is None:
            name_list = ["Input"]
            for layer_name, lyr in self.spiking_model.named_children():
                name_list.append(layer_name)

        print(name_list)
        if verbose:
            print("Comparing activations for {0}".format(name_list))

        # Calculate activations for the torch analog model
        if compute_rate:
            tsrAnalogData = data.mean(0).unsqueeze(0)
        else:
            tsrAnalogData = data.sum(0).unsqueeze(0)
        with torch.no_grad():
            analog_activations = get_activations(
                self.analog_model, tsrAnalogData, name_list=name_list
            )

        # Calculate activations for spiking model
        spike_rates = get_network_activations(
            self.spiking_model, data, name_list=name_list, bRate=compute_rate
        )
        return analog_activations, spike_rates

    def plot_comparison(
        self,
        data,
        name_list: Optional[ArrayLike] = None,
        compute_rate=False,
    ):
        """
        Plots a scatter plot of all the activations

        :param data: Data to be processed
        :param name_list: ArrayLike with names of all the layers of interest to be compared
        :param compute_rate: Compare firing rates instead of spike count
        """
        import pylab

        if name_list is None:
            name_list = ["Input"]
            for layer_name, lyr in self.spiking_model.named_children():
                name_list.append(layer_name)

        analog_activations, spike_rates = self.compare_activations(
            data, name_list=name_list, compute_rate=compute_rate,
        )
        for nLyrIdx in range(len(name_list)):
            pylab.scatter(
                spike_rates[nLyrIdx],
                analog_activations[nLyrIdx],
                label=name_list[nLyrIdx],
            )
        if compute_rate:
            pylab.xlabel("Spike rates (Hz)")
        else:
            pylab.xlabel("# Spike count")
        pylab.ylabel("Analog activations")
        pylab.legend()
        return analog_activations, spike_rates

    def reset_states(self):
        """
        Reset all neuron states in the submodules
        """
        for lyr in self.modules():
            if isinstance(lyr, SpikingLayer):
                lyr.reset_states()

    def get_synops(self, num_evs_in=None) -> pd.DataFrame:
        if num_evs_in is not None:
            warnings.warn("num_evs_in is deprecated and has no effect")

        SynOps_dataframe = pd.DataFrame()
        for (layer_name, lyr) in self.spiking_model.named_modules():
            if hasattr(lyr, 'synops'):
                SynOps_dataframe = SynOps_dataframe.append(
                    pd.Series(
                        {
                            "Layer": layer_name,
                            "In": lyr.tot_in,
                            "Fanout_Prev": lyr.fanout,
                            "SynOps": lyr.synops,
                            "Time_window": lyr.tw,
                            "SynOps/s": lyr.synops / lyr.tw * 1000,
                        }
                    ),
                    ignore_index=True,
                )
        SynOps_dataframe.set_index("Layer", inplace=True)
        return SynOps_dataframe
