from warnings import warn

import torch.nn as nn
from sinabs import Network
import sinabs.layers as sl
from typing import Dict, Union
import samna

import speckdemo as sd


def to_speck_config(snn: Union[nn.Module, sl.TorchLayer]) -> Dict:
    """
    Build a configuration object of a given module

    :param snn: sinabs.Network or sinabs.layers.TorchLayer instance
    """
    config = sd.configuration.SpeckConfiguration()

    if isinstance(snn, Network):
        for layer in snn.spiking_model.children():
            config[layer.layer_name] = to_speck_config(layer)
            # Populate source/destination layers here
            # Consolidate pooling and conv layers here
    elif isinstance(snn, sl.TorchLayer):
        # TODO: Do your thing for config
        ...
    elif isinstance(snn, sl.SpikingConv2dLayer):

        # TODO: Need info about next layer type
        destinations = (...,)
        layer_idx = ...

        layer_config = spiking_conv2d_to_speck(snn)
        speck_layer = config.cnn_layers[layer_idx]
        speck_layer.set_dimensions(layer_config["dimensions"])
        speck_layer.set_weights(layer_config["weights"])
        speck_layer.set_biases(layer_config["biases"])
        for param, value in layer_config["layer_params"]:
            setattr(speck_layer, param, value)

    return config


def spiking_conv2d_to_speck(layer):
    summary = layer.summary()
    dimensions = sd.configuration.CNNLayerDimensions()

    # - Padding
    padding_x, padding_y = summary["Padding"][0], summary["Padding"][2]
    if padding_x != summary["Padding"][1]:
        warn(
            "Left and right padding must be the same. "
            + "Will ignore value provided for right padding."
        )
    if padding_y != summary["Padding"][3]:
        warn(
            "Top and bottom padding must be the same. "
            + "Will ignore value provided for bottom padding."
        )
    if not 0 <= padding_x < 8:
        padding_x = max(min(padding_x, 7), 0)
        warn(
            "Horizontal padding must be between 0 and 7. "
            + "Values outside of this range are clipped."
        )
    if not 0 <= padding_y < 8:
        padding_y = max(min(padding_y, 7), 0)
        warn(
            "Vertical padding must be between 0 and 7. "
            + "Values outside of this range are clipped."
        )
    dimensions.padding.x = padding_x
    dimensions.padding.y = padding_y

    # - Stride
    stride_y, stride_x = summary["Stride"]
    if stride_x not in [1, 2, 4, 8]:
        raise ValueError("Horizontal stride must be in [1, 2, 4, 8].")
    if stride_y not in [1, 2, 4, 8]:
        raise ValueError("Vertical stride must be in [1, 2, 4, 8].")
    dimensions.x = stride_x
    dimensions.y = stride_y

    # - Kernel size
    kernel_size = summary["Kernel"][0]
    if kernel_size != summary["Kernel"][1]:
        warn(
            "Width and height of kernel must be the same. "
            + "Will use value provided for width."
        )
    if not 1 <= kernel_size < 17:
        kernel_size = max(min(kernel_size, 16), 1)
        warn(
            "Kernel size must be between 1 and 16. "
            + "Values outside of this range are clipped."
        )
    dimensions.kernel_size = kernel_size

    # - Input and output shapes
    dimensions.input_shape.feature_count = summary["Input_Shape"][0]
    dimensions.input_shape.size.y = summary["Input_Shape"][1]
    dimensions.input_shape.size.x = summary["Input_Shape"][2]
    dimensions.output_shape.feature_count = summary["Output_Shape"][0]
    dimensions.output_shape.size.y = summary["Output_Shape"][1]
    dimensions.output_shape.size.x = summary["Output_Shape"][2]
    # Is output shape not dependent on other values? Why can it be set?
    # Maybe better to pass dict instead of dimensions object

    # - Weights and biases
    weights, biases = layer.parameters()
    if not layer.bias:
        biases *= 0
    weights = weights.tolist()
    biases = biases.tolist()

    # - Neuron states
    neuron_states = layer.state.tolist()

    # - Resetting vs returning to 0
    return_to_zero = layer.membrane_subtract is not None
    if return_to_zero and layer.membrane_reset != 0:
        warn("Resetting of membrane potential is always to 0.")
    elif (not return_to_zero) and layer.membrane_subtract != layer.threshold_high:
        warn("Subtraction of membrane potential is always by high threshold.")

    layer_params = dict(
        return_to_zero=return_to_zero,
        threshold_high=layer.threshold_high,
        threshold_low=layer.threshold_low,
        monitor_enable=...,  # What is this?
        leak_enable=...,  # Leak in sinabs?
    )

    return {
        "layer_params": layer_params,
        "dimensions": dimensions,
        "weights": weights,
        "biases": biases,
        "neuron_states": neuron_states,
    }


def write_to_device(config: Dict, device: samna.SpeckModel, weights=None):
    """
    Write your model configuration to dict

    :param config:
    :param device:
    :return:
    """
    device.set_config(to_speck_config(config))
    if weights:
        device.set_weights(weights)
    device.apply()


def to_speck_config(config: Dict) -> samna.SpeckConfig:
    speck_config = samna.SpeckConfig()
    # TODO

    # Populate the config
    return speck_config
