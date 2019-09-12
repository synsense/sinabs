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

import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from .layers import TorchLayer
from typing import Iterable, List


def get_keras_activations(model, inp, name_list=None):
    """
    :param model: Keras model
    :param inp:  input data to be processed by keras model
    :param name_list: list of all layers whose activations need to be compared
    """
    from tensorflow import keras

    # Generate default list of layers
    if name_list is None:
        name_list = [lyr.name for lyr in model.layers]
        # Add input in the list of layer names if not explicitly defined as a layer name
        if (
            type(model.layers[0]) is keras.layers.Input
            or type(model.layers[0]) is keras.layers.InputLayer
        ):
            pass
        else:
            name_list = ["Input"] + name_list

    activations = []
    # Extract activity for each layer of interest
    for layer_name in name_list:
        if layer_name == "Input":
            # Bypass input layers
            activations.append(inp)
            continue
        lyr = model.get_layer(layer_name)
        if type(lyr) is keras.layers.Input or type(lyr) is keras.layers.InputLayer:
            activations.append(inp)
            continue
        # Bypass Flattening layer
        if type(lyr) is keras.layers.Flatten:
            continue
        # Bypass Dropout layer
        if type(lyr) is keras.layers.Dropout:
            continue

        modelx = keras.Model(model.input, lyr.output)
        activations.append(modelx.predict(inp))
        del (modelx)

    assert len(name_list) == len(activations)

    return activations


def get_activations(torchanalog_model, tsrData, name_list=None):
    """
    Return torch analog model activations for the specified layers
    """
    torch_modules = dict(torchanalog_model.named_modules())

    # Populate layer names
    if name_list is None:
        name_list = ["Input"] + list(torch_modules.keys())[1:]

    analog_activations = []

    for layer_name in name_list:
        if layer_name == "Input":
            # Bypass input layers
            analog_activations.append(tsrData)

    # Define hook
    def hook(module, inp, output):
        arrOut = output.detach().cpu().numpy()
        analog_activations.append(arrOut)

    hooklist = []
    # Attach and register hook
    for layer_name in name_list:
        if layer_name == "Input":
            continue
        torch_layer = torch_modules[layer_name]
        hooklist.append(torch_layer.register_forward_hook(hook))

    # Do a forward pass
    with torch.no_grad():
        torchanalog_model.eval()
        torchanalog_model(tsrData)

    # Remove hooks
    for h in hooklist:
        h.remove()

    return analog_activations


def get_network_activations(
    model: nn.Module, inp, name_list: List = None, bRate: bool = False
) -> [np.ndarray]:
    """
    Returns the activity of neurons in each layer of the network

    :param model: Model for which the activations are to be read out
    :param inp: Input to the model
    :param bRate: If true returns the rate, else returns spike count
    :param name_list: list of all layers whose activations need to be compared
    """
    spike_counts = []
    tSim = len(inp)

    # Define hook
    def hook(module, inp, output):
        arrOut = output.float().sum(0).cpu().numpy()
        spike_counts.append(arrOut)

    # Generate default list of layers
    if name_list is None:
        name_list = ["Input"] + [lyr.layer_name for lyr in model.layers]

    # Extract activity for each layer of interest
    for layer_name in name_list:
        # Append input activity
        if layer_name == "Input":
            spike_counts.append(inp.float().sum(0).cpu().numpy() * 1000)
        else:
            # Activity of other layers
            lyr = dict(model.named_modules())[layer_name]
            lyr.register_forward_hook(hook)

    with torch.no_grad():
        model(inp)

    if bRate:
        spike_counts = [(counts / tSim * 1000) for counts in spike_counts]
    return spike_counts


def summary(model: TorchLayer) -> pd.DataFrame:
    """
    This method returns the summary of a model
    :param model:
    :return: Pandas DataFrame
    """
    summary_dataframe = pd.DataFrame()
    for layer_name, lyr in model.named_children():
        try:
            summary_dataframe = summary_dataframe.append(
                lyr.summary(), ignore_index=True
            )
        except AttributeError:
            if lyr.__class__.__name__ == "ZeroPad2d":
                pass
            elif lyr.__class__.__name__ == "Sequential":
                summary_dataframe = summary_dataframe.append(
                    summary(lyr), ignore_index=True
                )
            else:
                raise Exception("Unknown layer type {0}".format(lyr.__class__.__name__))

    summary_dataframe.set_index("Layer", inplace=True)
    return summary_dataframe


def search_parameter(
    keys: Iterable, layer_name: str, strEndsWith: str = "weight"
) -> str:
    """
    Return the key that corresponds to a layer name and its weights.
    This is an internal convenience method and not intended for generic searches
    :param keys: List of keys to search `layer_name`
    :param layer_name: String to search at the start of the strings in keys
    :param strEndsWith: weight / bias
    :return: Returns the string that satisfies the requirements
    """
    dot = "."
    all_key_matches = []
    key: str
    for key in keys:
        if key.startswith(layer_name + dot) and key.endswith(dot + strEndsWith):
            all_key_matches.append(key)
    try:
        assert len(all_key_matches) is 1
        return all_key_matches[0]
    except AssertionError:
        raise Exception(
            "Unique key {2} not found in {1}: {0}".format(
                all_key_matches, keys, layer_name
            )
        )
