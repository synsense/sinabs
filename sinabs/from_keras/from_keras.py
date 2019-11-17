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

import json
from collections import OrderedDict

from sinabs.layers import *
import torch.nn as nn
import pandas as pd
from sinabs.cnnutils import infer_output_shape, compute_padding
import numpy as np
from typing import Optional, Union, List, Tuple, Dict

from sinabs.layers import FlattenLayer, SpikingConv2dLayer, QuantizeLayer, InputLayer, SumPooling2dLayer, \
    ZeroPad2dLayer, SpikingMaxPooling2dLayer
from sinabs.network import Network

ArrayLike = Union[np.ndarray, List, Tuple]


def from_json(
    keras_json: str,
    input_shape: Optional[ArrayLike] = None,
    quantize_analog_activation: bool = False,
    network: Network = None,
) -> Network:
    """
    Initialize the object from a keras model's json string

    :param keras_json: str Initialize `Network` from keras json string
    :param input_shape: Tuple shape of input
    :param quantize_analog_activation: True if analog layer's activations are to be quantized
    :param spiking_model: optional Network object to initialize
    :return: :class:`.network.Network`
    """
    keras_config = json.loads(keras_json)
    if network is None:
        model = Network(
            input_shape=input_shape, quantize_activation=quantize_analog_activation
        )
    else:
        model = network

    model.spiking_model = from_model_keras_config(
        keras_config, input_shape=input_shape, spiking=True
    )
    model.analog_model = from_model_keras_config(
        keras_config,
        input_shape=input_shape,
        spiking=False,
        quantize_analog_activation=quantize_analog_activation,
    )
    model.graph = extract_graph_from_keras_config(keras_config)
    return model


def from_model(
    keras_model,
    input_shape: Optional[ArrayLike] = None,
    quantize_activation: bool = False,
    nbit_quantize: Optional[int] = None,
    network=None,
) -> Network:
    """
    Initialize the object from a keras model

    :param keras_model: Initialize `Network` from keras model (object)
    :param input_shape: Tuple shape of input
    :param quantize_activation: True if analog layer's activations are to be quantized
    :param spiking_model: optional Network object to initialize
    :return: :class:`.network.Network`
    """

    if network is None:
        model = Network(
            input_shape=input_shape, quantize_activation=quantize_activation
        )
    else:
        model = network

    keras_json: str = keras_model.to_json()
    kerasWeights = keras_model.get_weights()
    data_format = infer_data_format(keras_model.get_config())

    from_json(
        keras_json,
        input_shape=input_shape,
        quantize_analog_activation=quantize_activation,
        network=model,
    )
    model.set_weights(
        weights=kerasWeights,
        is_from_keras=True,
        img_data_format=data_format,
        nbit_quantize=nbit_quantize,
        auto_rescale=True,
    )
    model.keras_model = keras_model

    return model


def infer_data_format(keras_config: dict) -> str:
    """
    Infer the data format of the model (channels_first/channels_last) from keras configuration

    :param keras_config: keras configuration dictionary
    :return: str 'channels_first' or 'channels_last' (If none is inferred, channels_first is returned)
    """
    data_formats = all_data_formats(keras_config)
    try:
        assert len(set(data_formats)) <= 1
    except AssertionError:
        raise ValueError(
            "Inconsistent `data_format` across layers {0}".format(data_formats)
        )

    if len(data_formats) >= 1:
        return data_formats[0]
    else:
        return "channels_first"


def all_data_formats(keras_config: dict) -> List:
    """
    Recursivly search for "data_format" in keras configuration

    :param keras_config: keras configuration dictionary
    :return: list -- the list of data_format
    """
    key = "data_format"
    data_format_list = []
    if hasattr(keras_config, "items"):
        for k, v in keras_config.items():
            if k == key:
                data_format_list.append(v)
            if isinstance(v, dict):
                data_format_list += all_data_formats(v)
            elif isinstance(v, list):
                for data in v:
                    data_format_list += all_data_formats(data)
    return data_format_list


def from_layer_keras_conf(
    layer_config, input_shape: Tuple, spiking=False, quantize_analog_activation=False
):
    """
    Load a layer from its keras conf, any of the supported layers can be initialized with this method

    :param layer_config: keras layer configuration
    :param input_shape: input shape
    :param spiking: bool True if a spiking layer is to be created
    :param quantize_analog_activation: True if analog layer's activations are to be quantized
    :return: [(layer_name, nn.Module)] Returns a list of layers and their names
    """
    if layer_config["class_name"] in ["InputLayer", "Input"]:
        layer_list = from_input_keras_conf(layer_config, input_shape, spiking=False)
    elif layer_config["class_name"] == "Cropping2D":
        layer_list = from_cropping2d_keras_conf(
            layer_config, input_shape, spiking=spiking
        )
    elif layer_config["class_name"] == "ZeroPadding2D":
        layer_list = from_zeropad2d_keras_conf(
            layer_config, input_shape, spiking=spiking
        )
    elif layer_config["class_name"] == "Flatten":
        layer_list = from_flatten_keras_conf(layer_config, input_shape, spiking=spiking)
    elif layer_config["class_name"] == "Dense":
        layer_list = from_dense_keras_conf(
            layer_config,
            input_shape,
            spiking=spiking,
            quantize_analog_activation=quantize_analog_activation,
        )
    elif layer_config["class_name"] == "Conv2D":
        layer_list = from_conv2d_keras_conf(
            layer_config,
            input_shape,
            spiking=spiking,
            quantize_analog_activation=quantize_analog_activation,
        )
    elif layer_config["class_name"] == "AveragePooling2D":
        layer_list = from_avgpool2d_keras_conf(
            layer_config, input_shape, spiking=spiking
        )
    elif layer_config["class_name"] == "MaxPooling2D":
        layer_list = from_maxpool2d_keras_conf(
            layer_config, input_shape, spiking=spiking
        )
    elif layer_config["class_name"] == "Dropout":
        layer_list = from_dropout_keras_conf(layer_config, input_shape, spiking=spiking)
    else:
        raise Exception("Unknown layer type {0}".format(layer_config))
    return layer_list


def from_model_keras_config(
    keras_config,
    input_shape: tuple = None,
    spiking=False,
    quantize_analog_activation=False,
) -> nn.Module:
    """
    Create a model from keras config

    :param keras_config: keras model configuration
    :param input_shape: Optional input shape, if left unspecified, this will be inferred from model's InputLayer
    :param spiking: bool True if a spiking model is to be created
    :param quantize_analog_activation: True if analog layer's activations are to be quantized
    :return: :class:`.network.Network`
    """
    if keras_config["class_name"] == "Sequential":
        sequential = True
    else:
        sequential = False

    keras_layers = keras_config["config"]["layers"]
    # Initialize containers for layers
    layer_list = []

    # Determine input dimensions
    inputInferred = None
    for nIdx, layer_config in enumerate(keras_layers):
        dataFormat = infer_data_format(keras_config)
        if layer_config["class_name"] in ["InputLayer", "Input"]:
            inputInferred = get_input_shape_from_keras_conf(layer_config=layer_config)
            if dataFormat == "channels_last":
                inputInferred = (inputInferred[-1], *inputInferred[:-1])

    if inputInferred:
        if input_shape:
            if inputInferred != inputInferred:
                raise AssertionError(
                    "Specified input shape {0} do not match that of the model {1}".format(
                        input_shape, inputInferred
                    )
                )
        else:
            input_shape = inputInferred
    else:
        if input_shape is None:
            raise Exception(
                "Input shape needs to be specified, either in the model or in the function call."
            )

    name_list = []
    output_shapePrev = input_shape
    # Build each of the layers
    for nIdx, layer_config in enumerate(keras_layers):
        # Append layer summary
        name_list.append(layer_config["config"]["name"])
        lyrs = from_layer_keras_conf(
            layer_config,
            output_shapePrev,
            spiking=spiking,
            quantize_analog_activation=quantize_analog_activation,
        )
        if len(lyrs):
            layer_list += lyrs
            try:
                output_shapePrev = lyrs[-1][1].output_shape
            except AttributeError as e:
                output_shapePrev = infer_output_shape(lyrs[-1][1], output_shapePrev)

    # Build map if not sequential
    if sequential:
        connectivity_map = None
    else:
        connectivity_map = extract_graph_from_keras_config(keras_config)
    # Build a model given layers
    model = build_model(layer_list, connectivity_map, sequential)

    return model


def extract_graph_from_keras_config(keras_config) -> pd.DataFrame:
    """
    Generate a boolean map of connectivity [source, destination] = True

    :param keras_config: Keras Configuration to readout network topology from
    :return: pandas DataFrame with indices and columns being layer names
    """
    sequential = False
    if keras_config["class_name"] == "Sequential":
        sequential = True

    keras_layers = keras_config["config"]["layers"]
    name_list = []
    for layer_config in keras_layers:
        name_list.append(layer_config["config"]["name"])

    connectivity_map = np.zeros(2 * (len(keras_layers),)).astype(bool)

    for nIdx, layer_config in enumerate(keras_layers):
        # Determine inbound layers
        if not sequential:
            vInboundLayers = [v[0][0] for v in layer_config["inbound_nodes"]]
            vInbountLayerIdxs = where(name_list, vInboundLayers)
        else:
            vInbountLayerIdxs = nIdx - 1
        connectivity_map[vInbountLayerIdxs, nIdx] = True
    return pd.DataFrame(connectivity_map, columns=name_list, index=name_list)


def build_model(
    layer_list: List, connectivity_map: Optional[np.ndarray], sequential=False
) -> nn.Module:
    """
    Build a torch module with the given layers

    :param layer_list: A list of all layers as (layer_name, torch_layer)
    :param connectivity_map: Optional, boolean map of connectivity [source, destination] = True
    :param sequential: If True, a sequential model is created ignoring the connectivity map
    :return: Torch Module created with the layers
    """
    if sequential:
        if len(layer_list) == 1:
            layer_name, model = layer_list[0]
        elif len(layer_list) > 1:
            model = nn.Sequential(OrderedDict(layer_list))
        else:
            model = nn.Sequential()
    else:
        # TODO: For now we only create sequential models
        model = nn.Sequential(OrderedDict(layer_list))

    return model


def where(all_layer_names_list: List, name_list: List) -> List:
    """
    Convenience function to find index of a value or a list of values in a list

    :param all_layer_names_list: The list to search in
    :param name_list: The list of values to search
    :return: Returns a list of indices of the original list where the values
    """
    pos = []
    for lyrName in name_list:
        try:
            pos.append(all_layer_names_list.index(lyrName))
        except ValueError:
            raise Exception("Inbound layer couldn't be found")
    return pos


def extract_json_weights(strModelFile: str, strPath: str = "./"):

    """
    From a keras .h5 model file, generate a json file and a numpy weight file

    :param strModelFile: str Model file name.
    :param strPath: str Path to save the model json file and weight files. The json file is saved to `strPath/models/` and weights to `strPath/weights/`
    """

    import tensorflow as tf
    import os

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    from tensorflow import keras
    import numpy as np

    keras_model = keras.models.load_model(strModelFile)
    model_name = strModelFile[:-3].split("/")[-1]
    # Save model file
    if not os.path.isdir(strPath + "/models/"):
        os.makedirs(strPath + "/models/")
    with open(strPath + "/models/" + model_name + ".json", "w") as jsonFile:
        jsonFile.write(keras_model.to_json())

    # Save weights of the model
    if not os.path.isdir(strPath + "/weights/"):
        os.makedirs(strPath + "/weights/")
    model_weights = keras_model.get_weights()
    np.savez(strPath + "/weights/" + model_name, *model_weights)


def transposeKeras2Torch(weights: List) -> List:
    """
    Transpose the shape of keras weights from channels_last format of weights to channels first style
    """
    reshaped_weights = []
    for w in weights:
        if len(w.shape) == 4:
            # Convolutional weights
            reshaped_w = w.transpose(3, 2, 0, 1).astype(float)
            reshaped_weights.append(reshaped_w)
        elif len(w.shape) == 2:
            # Dense layer weights
            reshaped_w = w.transpose(1, 0)
            reshaped_weights.append(reshaped_w)
        elif len(w.shape) == 1:
            # Bias, do nothing
            reshaped_weights.append(w)
    return reshaped_weights


def from_cropping2d_keras_conf(
        layer_config: dict, input_shape: ArrayLike, spiking=False
) -> List:
    """
    Load cropping 2d layer from Json configuration

    :param layer_config: keras configuration dictionary for this object
    :param input_shape: input data shape to determine output dimensions (channels, height, width)
    :param spiking: bool, is irrelevant for cropping layer. Only here for consistency
    :return: [(layer_name, nn.Module)] Returns a list of layers and their names
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]

    # Determine output dims
    cropping = layer_config["config"]["cropping"]

    # Initialize cropping layer
    torch_layer = Cropping2dLayer(
        image_shape=input_shape[1:], cropping=cropping, layer_name=layer_name
    )

    # Set input shape
    torch_layer.input_shape = input_shape

    return [(layer_name, torch_layer)]


def from_flatten_keras_conf(
    layer_config: dict, input_shape: ArrayLike, spiking=False
) -> [(str, nn.Module)]:
    """
    Load Flatten layer from keras configuration
    :param layer_config: configuration dictionary
    :param input_shape: input data shape to determine output dimensions
    :param spiking: bool True if spiking layer needs to be loaded
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]

    if spiking:
        # For a spiking layer, the next convolutional layer automatically re-formats the data structure
        return []
    else:
        # Determine output dims
        torch_layer = FlattenLayer(input_shape=input_shape, layer_name=layer_name)
        torch_layer.input_shape = input_shape

    return [(layer_name, torch_layer)]


def from_conv2d_keras_conf(
    layer_config: dict,
    input_shape: ArrayLike,
    spiking: bool = False,
    quantize_analog_activation: bool = False,
) -> List:
    """
    Load Convolutional layer from Json configuration

    :param layer_config: keras configuration dictionary for this object
    :param input_shape: input data shape to determine output dimensions (channels, height, width)
    :param spiking: bool True if spiking layer needs to be loaded
    :param quantize_analog_activation: Whether or not to add a quantization layer for the analog model
    :return: [(layer_name, nn.Module)] Returns a list of layers and their names
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    # Extract layer name
    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    layer_list = []

    channels, height, width = input_shape

    kernel_shape = layer_config["config"]["kernel_size"]
    pad_mod = layer_config["config"]["padding"]
    vStride = layer_config["config"]["strides"]

    # Padding
    if pad_mod == "valid":
        padding = (0, 0, 0, 0)
    else:
        # Compute padding
        padding = compute_padding(kernel_shape, input_shape, pad_mod)

    # Create layers
    if spiking:
        torch_spiking_conv2d = SpikingConv2dLayer(
            channels_in=channels,
            image_shape=input_shape[-2:],
            channels_out=layer_config["config"]["filters"],
            kernel_shape=kernel_shape,
            strides=layer_config["config"]["strides"],
            padding=padding,
            bias=layer_config["config"]["use_bias"],
            threshold=1.0,
            threshold_low=-1.0,
            membrane_subtract=1.0,
            layer_name=layer_name,
        )

        layer_list.append((layer_name, torch_spiking_conv2d))
    else:
        # Create a padding layer
        pad_layer = nn.ZeroPad2d(padding)
        layer_list.append((layer_name + "_padding", pad_layer))
        # Create a convolutional layer
        analog_layer = nn.Conv2d(
            channels,
            layer_config["config"]["filters"],
            kernel_shape,
            stride=vStride,
            bias=layer_config["config"]["use_bias"],
        )
        layer_list.append((layer_name + "_conv", analog_layer))
        # Activation
        if quantize_analog_activation:
            activation_layer_name = (
                layer_name + "_" + layer_config["config"]["activation"]
            )
        else:
            activation_layer_name = layer_name

        if layer_config["config"]["activation"] == "linear":
            pass
        elif layer_config["config"]["activation"] == "relu":
            analog_activation_layer = nn.ReLU()
            layer_list.append((activation_layer_name, analog_activation_layer))
        elif layer_config["config"]["activation"] == "sigmoid":
            analog_activation_layer = nn.Sigmoid()
            layer_list.append((activation_layer_name, analog_activation_layer))
        elif layer_config["config"]["activation"] == "softmax":
            analog_activation_layer = nn.ReLU()
            layer_list.append((activation_layer_name, analog_activation_layer))
        else:
            raise NotImplementedError

        # Create a Quantization layer
        if quantize_analog_activation:
            torch_quantize_layer = QuantizeLayer()
            layer_list.append((layer_name, torch_quantize_layer))

    if len(layer_list) > 1:
        return [(layer_name, nn.Sequential(OrderedDict(layer_list)))]
    else:
        return layer_list


def from_dense_keras_conf(
    layer_config: Dict,
    input_shape: Tuple,
    spiking=False,
    quantize_analog_activation=False,
) -> List:
    """
    Create a Dense layer from keras configuration

    :param layer_config: keras layer configuration
    :param input_shape: input shape
    :param spiking: bool True if a spiking layer is to be created
    :param quantize_analog_activation: True if analog layer's activations are to be quantized
    :return: [(layer_name, nn.Module)] Returns a list of layers and their names
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    layer_list = []

    if spiking:
        channels, height, width = input_shape
        # Initialize convolutional layer
        torch_spiking_conv2d = SpikingConv2dLayer(
            channels_in=channels,
            image_shape=input_shape[-2:],
            channels_out=layer_config["config"]["units"],
            kernel_shape=(height, width),
            strides=(height, width),
            padding=(0, 0, 0, 0),
            bias=layer_config["config"]["use_bias"],
            threshold=1.0,
            threshold_low=-1.0,
            membrane_subtract=1.0,
            layer_name=layer_name,
        )
        torch_spiking_conv2d.input_shape = input_shape
        layer_list.append((layer_name, torch_spiking_conv2d))
    else:
        # Input should have already been flattened
        try:
            nInputLength, = input_shape
        except ValueError as e:
            raise Exception(
                "Input shape of a Dense layer should be 1 dimensional (per batch), use Flatten"
            )
        nOutputLength = layer_config["config"]["units"]
        torch_analogue_layer = nn.Linear(
            nInputLength, nOutputLength, bias=layer_config["config"]["use_bias"]
        )
        layer_list.append((layer_name + "_Linear", torch_analogue_layer))

        if quantize_analog_activation:
            layer_nameActivation = (
                layer_name + "_" + layer_config["config"]["activation"]
            )
        else:
            layer_nameActivation = layer_name

        # Activation
        if layer_config["config"]["activation"] == "linear":
            pass
        elif layer_config["config"]["activation"] == "relu":
            torch_analogue_layerActivation = nn.ReLU()
            layer_list.append((layer_nameActivation, torch_analogue_layerActivation))
        elif layer_config["config"]["activation"] == "sigmoid":
            torch_analogue_layerActivation = nn.Sigmoid()
            layer_list.append((layer_nameActivation, torch_analogue_layerActivation))
        elif layer_config["config"]["activation"] == "softmax":
            torch_analogue_layerActivation = nn.ReLU()
            layer_list.append((layer_nameActivation, torch_analogue_layerActivation))
        else:
            raise NotImplementedError

        if quantize_analog_activation:
            torch_quantize_layer = QuantizeLayer()
            layer_list.append((layer_name, torch_quantize_layer))

    if len(layer_list) > 1:
        return [(layer_name, nn.Sequential(OrderedDict(layer_list)))]
    else:
        return layer_list


def get_input_shape_from_keras_conf(layer_config) -> Tuple:
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    input_shape = layer_config["config"]["batch_input_shape"]
    # Input dimensions ignoring batch information
    return input_shape[1:]


def from_input_keras_conf(
    layer_config, input_shape=None, spiking=False
) -> [(str, nn.Module)]:
    """
    Load input layer from Json configuration
    :param layer_config: configuration dictionary
    :param input_shape: This parameter is here only for API consistency
    :param spiking: bool True if spiking layer needs to be loaded
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]

    layer_list = [
        (layer_name, InputLayer(input_shape=input_shape, layer_name=layer_name))
    ]
    return layer_list


def from_sumpool2d_keras_conf(
    layer_config, input_shape: Tuple, spiking: bool = False
) -> [(list, nn.Module)]:
    """
    Creates AveragePooling layer from keras configuration file
    This is actually a proxy to SumPooling2dLayer in case of spiking version

    :param layer_config:
    :param input_shape:
    :param spiking: The spiking and non spiking layers act the same for sum pooling, so this option is ignored
    :return: [(layer_name, nn.Module)] Returns a list of layers and their names
    """
    return from_avgpool2d_keras_conf(layer_config, input_shape, spiking=True)


def from_avgpool2d_keras_conf(
    layer_config, input_shape: Tuple, spiking: bool = False
) -> [(list, nn.Module)]:
    """
    Crete a Average pooling layer

    :param layer_config:
    :param input_shape:
    :param spiking:
    :return:
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    layer_list = []
    pool_size = layer_config["config"]["pool_size"]
    strides = layer_config["config"]["strides"]
    pad_mod = layer_config["config"]["padding"]

    if pad_mod == "valid":
        padding = (0, 0, 0, 0)
    else:
        # Compute padding
        padding = compute_padding(pool_size, input_shape, pad_mod)

    if spiking:
        spike_pool = SumPooling2dLayer(
            image_shape=input_shape[1:],
            pool_size=pool_size,
            padding=padding,
            strides=strides,
            layer_name=layer_name,
        )
        spike_pool.input_shape = input_shape
        layer_list.append((layer_name, spike_pool))
    else:
        # Create a padding layer
        if padding != (0, 0, 0, 0):
            torch_layerPad = nn.ZeroPad2d(padding)
            layer_list.append((layer_name + "_padding", torch_layerPad))
        # Pooling layer initialization
        analogue_pool = nn.AvgPool2d(kernel_size=pool_size, stride=strides)
        layer_list.append((layer_name, analogue_pool))

    if len(layer_list) > 1:
        return [(layer_name, nn.Sequential(OrderedDict(layer_list)))]
    else:
        return layer_list


def from_zeropad2d_keras_conf(
    layer_config, input_shape: Tuple, spiking: bool = False
) -> [(str, nn.Module)]:
    """
    Load ZeroPadding layer from Json configuration

    :param layer_config: configuration dictionary
    :param input_shape: input data shape to determine output dimensions
    :param spiking: bool True if spiking layer needs to be loaded (This parameter is immaterial for this layer)
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    # Determing output dims
    tb, lr = layer_config["config"]["padding"]

    torch_layer = ZeroPad2dLayer(
        image_shape=input_shape[1:], padding=(*lr, *tb), layer_name=layer_name
    )
    torch_layer.input_shape = input_shape
    return [(layer_name, torch_layer)]


def from_dropout_keras_conf(
    layer_config: Dict, input_shape: Tuple, spiking=False
) -> List:
    """
    Load Dropout  keras config file
    :param layer_config: configuration dictionary
    :param input_shape: input data shape to determine output dimensions
    :param spiking: bool True if spiking layer needs to be loaded
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    layer_list = []

    # Dropout rate
    dropout_rate = layer_config["config"]["rate"]

    if spiking:
        pass
    else:
        # Pooling layer initialization
        torch_analogue_layer = nn.Dropout(p=dropout_rate)
        layer_list.append((layer_name, torch_analogue_layer))

    return layer_list


def from_maxpool2d_keras_conf(
    layer_config, input_shape: Tuple, spiking: bool = False
) -> [(list, nn.Module)]:
    """
    Crete a Average pooling layer

    :param layer_config:
    :param input_shape:
    :param spiking:
    :return:
    """
    # Config depth consistency
    if "config" in layer_config:
        pass
    else:
        layer_config = {"config": layer_config}

    try:
        layer_name = layer_config["name"]
    except KeyError:
        layer_name = layer_config["config"]["name"]
    layer_list = []
    pool_size = layer_config["config"]["pool_size"]
    strides = layer_config["config"]["strides"]
    pad_mod = layer_config["config"]["padding"]

    if pad_mod == "valid":
        padding = (0, 0, 0, 0)
    else:
        # Compute padding
        padding = compute_padding(pool_size, input_shape, pad_mod)

    if spiking:
        spike_pool = SpikingMaxPooling2dLayer(
            image_shape=input_shape[1:],
            pool_size=pool_size,
            padding=padding,
            strides=strides,
            layer_name=layer_name,
        )
        spike_pool.input_shape = input_shape
        layer_list.append((layer_name, spike_pool))
    else:
        # Create a padding layer
        if padding != (0, 0, 0, 0):
            torch_layerPad = nn.ZeroPad2d(padding)
            layer_list.append((layer_name + "_padding", torch_layerPad))
        # Pooling layer initialization
        analogue_pool = nn.MaxPool2d(kernel_size=pool_size, stride=strides)
        layer_list.append((layer_name, analogue_pool))

    if len(layer_list) > 1:
        return [(layer_name, nn.Sequential(OrderedDict(layer_list)))]
    else:
        return layer_list