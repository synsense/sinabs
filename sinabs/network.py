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
from operator import mul
from functools import reduce
from typing import Optional, Union, List, Tuple
from .utils import (
    get_network_activations,
    get_activations,
    search_parameter,
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
        dummy_input = torch.zeros(
            [batch_size] + list(input_shape),
            requires_grad=False
        )
        self(dummy_input)

        [this_hook.remove() for this_hook in hook_list]

    def rescale_parameters(
        self,
        weight_rescaling: dict = None,
        bias_rescaling: dict = None,
        threshold_rescaling: dict = None,
        analog: bool = True,
        verbose: bool = False,
    ):
        """
        Rescale the weights and biases of the model

        :param bias_rescaling: dict of layername as key and multiplication factor as value
        :param weight_rescaling: dict of layername as key and multiplication factor as value
        :param threshold_rescaling: dict of layername as key and multiplication factor as value
        :param analog: bool to rescale analog model
        :param verbose: bool to print debugging messages
        """
        spiking_params = dict(self.spiking_model.named_parameters())
        analog_params = dict(self.analog_model.named_parameters())
        if weight_rescaling:
            for strLyr, scale in weight_rescaling.items():
                # Rescale spiking model parameters
                key = search_parameter(
                    spiking_params.keys(), strLyr, strEndsWith="weight"
                )
                param = spiking_params[key]
                if verbose:
                    print(
                        "Updating weights for parameter {0} of shape {1}".format(
                            key, param.shape
                        )
                    )
                param.data = param.data * scale
                # Rescale analog model parameters
                if analog:
                    key = search_parameter(
                        analog_params.keys(), strLyr, strEndsWith="weight"
                    )
                    paramAnalog = analog_params[key]
                    paramAnalog.data = paramAnalog.data * scale
        if bias_rescaling:
            for strLyr, scale in bias_rescaling.items():
                # Rescale biases for spiking model
                key = search_parameter(
                    spiking_params.keys(), strLyr, strEndsWith="bias"
                )
                param = spiking_params[key]
                param.data = param.data * scale
                # Rescale analog model parameters
                if analog:
                    key = search_parameter(
                        analog_params.keys(), strLyr, strEndsWith="bias"
                    )
                    param = analog_params[key]
                    param.data = param.data * scale
        if threshold_rescaling:
            for strLyr, scale in threshold_rescaling.items():
                # Rescale threshold
                lyr = dict(self.spiking_model.named_modules())[strLyr]
                lyr.threshold = lyr.threshold * scale
                lyr.threshold_low = lyr.threshold_low * scale
                lyr.membrane_subtract = lyr.membrane_subtract * scale

    def set_weights(
        self,
        weights: List,
        img_data_format: str = "channels_first",
        nbit_quantize: Optional[int] = None,
        auto_rescale: bool = False,
    ):
        """
        Load weights for this model from a list of numpy arrays

        :param weights: List of weight np.ndarrays
        :param img_data_format: str channels_first/channels_last input data format that the model has been trained with
        :param nbit_quantize: int Number of bits used to approximate the weights with
        :param auto_rescale: bool flag to auto rescale the weights for spiking model to compensate for average pooling
               sumpooling approximation
        """
        all_thresholds = {}
        # Quantize weights for nbit + 1 in total
        if nbit_quantize:
            nLyrCount = 0
            for nIndx in range(len(weights)):
                fWmax = np.max(np.abs(weights[nIndx]))
                fWscale = (2 ** nbit_quantize - 1) / fWmax
                weights[nIndx] = np.round(weights[nIndx] * fWscale) / fWscale
                fThreshold = np.round(fWscale)
                nBitThreshold = len(np.binary_repr(fThreshold.astype(int))) + 1
                print("%d bit is needed to represent threshold" % nBitThreshold)
                assert (
                    nBitThreshold <= 16
                ), "Number of bit of a threshold cannot be more than 16."
                fThreshold /= fWscale
                strLyrName = self.layers[nLyrCount][0]
                while not ("conv" in strLyrName or "dense" in strLyrName):
                    nLyrCount += 1
                    strLyrName = self.layers[nLyrCount][0]
                all_thresholds[strLyrName] = fThreshold
                nLyrCount += 1

        # Load weights for the analog model
        for indx, param in enumerate(self.analog_model.parameters()):
            assert param.data.shape == weights[indx].shape
            param.data = torch.from_numpy(weights[indx]).float()

        # Find layers that need to scale down the weights:
        all_weight_scale_factors = {}
        if auto_rescale:
            num_scaler = 1.0
            for strLyrName, lyr in self.layers:
                if "average_pooling" in strLyrName:
                    num_scaler *= reduce(mul, lyr.pool_size)
                elif "conv" in strLyrName or "dense" in strLyrName:
                    if num_scaler != 1:
                        all_weight_scale_factors[strLyrName] = 1.0 / num_scaler
                        num_scaler = 1

        # Load weights for spiking model
        assert len(weights) == len(list(self.spiking_model.parameters()))

        for indx, param in enumerate(self.spiking_model.parameters()):
            try:
                assert param.data.shape == weights[indx].shape
                param.data = torch.from_numpy(weights[indx]).float()
            except AssertionError:
                # Flatten + Dense layers to CNN weights
                if img_data_format == "channels_last":
                    w_reshaped = weights[indx].reshape(param.data.shape)
                    param.data = torch.from_numpy(w_reshaped).float()
                elif img_data_format == "channels_first":
                    if is_from_keras:

                        shape_new = np.array(param.shape)[[0, 2, 3, 1]]
                        w_reshaped = weights[indx].reshape(shape_new).astype(float)
                        w_reshaped = w_reshaped.transpose(0, 3, 1, 2)
                    else:
                        w_reshaped = weights[indx].reshape(param.data.shape)
                    param.data = torch.from_numpy(w_reshaped).float()
                    # Reshape weight for the analog model as well
                    weights[indx] = w_reshaped.reshape((len(w_reshaped), -1))

        # Load weights for the analog model
        for indx, param in enumerate(self.analog_model.parameters()):
            assert param.data.shape == weights[indx].shape
            param.data = torch.from_numpy(weights[indx]).float()

        # Rescale the parameters
        if all_weight_scale_factors:
            print(f"Scaling layer weights: {all_weight_scale_factors}")
        self.rescale_parameters(
            weight_rescaling=all_weight_scale_factors,
            threshold_rescaling=all_thresholds,
            analog=False,  # Only auto-rescale spiking model for average pooling.
        )

    def forward(self, tsrInput) -> torch.Tensor:
        """
        Forward pass for this model
        """
        with torch.no_grad():
            self.nEvsLastInput = tsrInput.sum().item()
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
            name_list = []
            for layer_name, lyr in self.layers:
                try:
                    name_list.append(lyr.layer_name)
                except AttributeError:
                    pass

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
            for layer_name, lyr in self.layers:
                try:
                    name_list.append(lyr.layer_name)
                except AttributeError:
                    pass

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
