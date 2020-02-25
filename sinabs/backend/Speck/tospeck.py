from warnings import warn

import torch.nn as nn
from sinabs import Network
import sinabs.layers as sl
from typing import Dict, List, Tuple, Union
import samna

import speckdemo as sd

## -- Parameters
SPECK_DVS_POOLING_SIZES = [1, 2, 4]
SPECK_CNN_POOLING_SIZES = [1, 2, 4, 8]
SPECK_CNN_STRIDE_SIZES = [1, 2, 4, 8]


def to_speck_config(snn: Union[nn.Module, sl.TorchLayer]) -> Dict:
    """
    Build a configuration object of a given module

    :param snn: sinabs.Network or sinabs.layers.TorchLayer instance
    """
    config = sd.configuration.SpeckConfiguration()

    if isinstance(snn, Network):
        layers = snn.spiking_model.children()

        i_layer = 0
        i_layer_speck = 0

        # - Iterate over layers from model
        while i_layer < len(layers):

            # Layer to be ported to Speck
            lyr_curr = layers[i_layer]

            if isinstance(lyr_curr, sl.SpikingConv2dLayer):
                # Object representing Speck layer
                speck_layer = config.cnn_layers[i_layer_speck]

                # Extract configuration specs from layer object
                layer_config = spiking_conv2d_to_speck(snn)
                # Update configuration of the Speck layer
                speck_layer.set_dimensions(layer_config["dimensions"])
                speck_layer.set_weights(layer_config["weights"])
                speck_layer.set_biases(layer_config["biases"])
                for param, value in layer_config["layer_params"]:
                    setattr(speck_layer, param, value)

                # - Consolidate pooling from subsequent layers
                pooling, i_next = consolidate_pooling(layers[i_layer + 1 :])

                # - Destination for CNN layer... make sure that is cnn or sum pooling?

                # For now: Sequential model, second destination always disabled
                speck_layer.destination[1].enable = False

                if i_next is not None:
                    # Set destination layer
                    speck_layer.destination[0].layer = i_layer_speck + 1
                    speck_layer.destination[0].pooling = pooling
                    speck_layer.destination[0].enable = True

                    # Add 1 to i_layer to go to next layer, + i_next for number
                    # of consolidated pooling layers
                    i_layer += 1 + i_next
                    i_layer_speck += 1

                else:
                    speck_layer.destination[0].enable = False
                    # TODO: How to route to readout layer? Does destination need to be set?
                    break

            elif isinstance(lyr_curr, sl.SumPooling2dLayer):
                # This case can only happen when `layers` starts with a pooling layer,
                # because all other pooling layers should get consolidated. Assume that
                # input then comes from DVS

                # Object representing Speck DVS
                dvs = config.dvs_layer
                pooling, i_next = consolidate_pooling(layers[i_layer:], dvs=True)
                dvs.pooling.y, dvs.pooling.x = pooling
                if i_next is not None:
                    dvs.destinations[0].layer = i_layer_speck
                    dvs.destinations[0].enable = True
                else:
                    break

        # TODO: Does anything need to be done after iterating over layers?
        print("Finished configuration of Speck.")

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


def consolidate_pooling(
    layers, dvs: bool = False
) -> Tuple[Union[int, Tuple[int], None], int]:
    """
    TODO: Handle case where resulting pooling would be too large
    consolidate_pooling - Consolidate the first `SumPooling2dLayer`s in `layers`
                          until the first object of different type.
    :param layers:  Iterable, containing `SumPooling2dLayer`s and other objects.
    :param dvs:     bool, if True, x- and y- pooling may be different and a
                          Tuple is returned instead of an integer.
    :return:
        int or tuple, consolidated pooling size. Tuple if `dvs` is `True`.
        int or None, index of first object in `layers` that is not a
                     `SumPooling2dLayer`, or `None`, if all objects in `layers`
                     are `SumPooling2dLayer`s.
    """

    pooling = 1 if dvs else (1, 1)

    for i_next, lyr in enumerate(layers):
        if isinstance(lyr, sl.SumPooling2dLayer):
            # Update pooling size
            new_pooling = get_sumpool2d_pooling_config(lyr)
            if dvs:
                pooling[0] *= new_pooling[0]
                pooling[1] *= new_pooling[1]
            else:
                pooling *= new_pooling
        else:
            return pooling, i_next

    # If this line is reached, all objects in `layers` are `SumPooling2dLayer`s.
    return pooling, None


def get_sumpool2d_pooling_config(layer, dvs: bool = True) -> Union[int, Tuple[int]]:
    summary = layer.summary()
    if any(pad != 0 for pad in summary["Padding"]):
        warn(
            f"SumPooling2dLayer `{layer.layer_name}`: Padding is not supported for pooling layers."
        )

    if dvs:
        pooling_y, pooling_x = summary["Pooling"]
        # Check pooling size
        if pooling_y not in SPECK_DVS_POOLING_SIZES:
            raise ValueError(
                f"SumPooling2dLayer `{layer.layer_name}`: Vertical pooling dimension for DVS must be in [1, 2, 4]."
            )
        if pooling_x not in SPECK_DVS_POOLING_SIZES:
            raise ValueError(
                f"SumPooling2dLayer `{layer.layer_name}`: Horizontal pooling dimension for DVS must be in [1, 2, 4]."
            )
        # Check whether pooling and strides match
        if summary["Stride"][0] != pooling_y or summary["Stride"][1] != pooling_x:
            raise ValueError(
                f"SumPooling2dLayer `{layer.layer_name}`: Stride size must be the same as pooling size."
            )
        return (pooling_y, pooling_x)
    else:
        pooling = summary["Pooling"][0]  # Is this the vertical dimension?
        # Check whether pooling is symmetric
        if pooling != summary["Pooling"][1]:
            raise ValueError(
                f"SumPooling2dLayer `{layer.layer_name}`: Pooling must be symmetric for CNN layers."
            )
        # Check whether pooling and strides match
        if any(stride != pooling for stride in summary["Stride"]):
            raise ValueError(
                f"SumPooling2dLayer `{layer.layer_name}`: Stride size must be the same as pooling size."
            )
        # Check pooling size
        if pooling not in SPECK_CNN_POOLING_SIZES:
            raise ValueError(
                f"SumPooling2dLayer `{layer.layer_name}`: Vertical pooling dimension for CNN layers must be in [1, 2, 4, 8]."
            )
        return pooling


def spiking_conv2d_to_speck(layer: sl.SpikingConv2dLayer) -> Dict:
    summary = layer.summary()
    dimensions = sd.configuration.CNNLayerDimensions()

    # - Padding
    padding_x, padding_y = summary["Padding"][0], summary["Padding"][2]
    if padding_x != summary["Padding"][1]:
        warn(
            f"SpikingConv2dLayer `{layer.layer_name}`: "
            + "Left and right padding must be the same. "
            + "Will ignore value provided for right padding."
        )
    if padding_y != summary["Padding"][3]:
        warn(
            f"SpikingConv2dLayer `{layer.layer_name}`: "
            + "Top and bottom padding must be the same. "
            + "Will ignore value provided for bottom padding."
        )
    if not 0 <= padding_x < 8:
        raise ValueError(
            f"SpikingConv2dLayer `{layer.layer_name}`: Horizontal padding must be between 0 and 7"
        )
    if not 0 <= padding_y < 8:
        raise ValueError(
            f"SpikingConv2dLayer `{layer.layer_name}`: Vertical padding must be between 0 and 7"
        )
    dimensions.padding.x = padding_x
    dimensions.padding.y = padding_y

    # - Stride
    stride_y, stride_x = summary["Stride"]
    if stride_x not in SPECK_CNN_STRIDE_SIZES:
        raise ValueError(
            f"SpikingConv2dLayer `{layer.layer_name}`: Horizontal stride must be in [1, 2, 4, 8]."
        )
    if stride_y not in SPECK_CNN_STRIDE_SIZES:
        raise ValueError(
            f"SpikingConv2dLayer `{layer.layer_name}`: Vertical stride must be in [1, 2, 4, 8]."
        )
    dimensions.stride.x = stride_x
    dimensions.stride.y = stride_y

    # - Kernel size
    kernel_size = summary["Kernel"][0]
    if kernel_size != summary["Kernel"][1]:
        raise ValueError(
            f"SpikingConv2dLayer `{layer.layer_name}`: Width and height of kernel must be the same."
        )
    if not 1 <= kernel_size < 17:
        kernel_size = max(min(kernel_size, 16), 1)
        raise ValueError(
            f"SpikingConv2dLayer `{layer.layer_name}`: Kernel size must be between 1 and 16."
        )
    dimensions.kernel_size = kernel_size

    # - Input and output shapes
    dimensions.input_shape.feature_count = summary["Input_Shape"][0]
    dimensions.input_shape.size.y = summary["Input_Shape"][1]
    dimensions.input_shape.size.x = summary["Input_Shape"][2]
    dimensions.output_shape.feature_count = summary["Output_Shape"][0]
    dimensions.output_shape.size.y = summary["Output_Shape"][1]
    dimensions.output_shape.size.x = summary["Output_Shape"][2]

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
        warn(
            f"SpikingConv2dLayer `{layer.layer_name}`: Resetting of membrane potential is always to 0."
        )
    elif (not return_to_zero) and layer.membrane_subtract != layer.threshold_high:
        warn(
            f"SpikingConv2dLayer `{layer.layer_name}`: Subtraction of membrane potential is always by high threshold."
        )

    layer_params = dict(
        return_to_zero=return_to_zero,
        threshold_high=layer.threshold_high,
        threshold_low=layer.threshold_low,
        monitor_enable=True,  # Yes or no?
        leak_enable=True,  # Or only if (bias != 0).any()?
    )

    return {
        "layer_params": layer_params,
        "dimensions": dimensions,
        "weights": weights,
        "biases": biases,
        "neuron_states": neuron_states,
    }


def identity_dimensions(input_shape: Tuple[int]) -> sd.configuration.CNNLayerDimensions:
    """
    identity_dimensions - Return `CNNLayerDimensions` for Speck such that the layer
                          performs an identity operation.
    :param input_shape:   Tuple with feature_count, vertical and horizontal size of
                          input to the layer.
    :return:
        CNNLayerDimensions corresponding to identity operation.
    """
    dimensions = sd.configuration.CNNLayerDimensions()
    # No padding
    dimensions.padding.x = 0
    dimensions.padding.y = 0
    # Stride 1
    dimensions.stride.x = 1
    dimensions.stride.y = 1
    # Input shape
    dimensions.input_shape.feature_count = input_shape[0]
    dimensions.input_shape.y = input_shape[1]
    dimensions.input_shape.x = input_shape[2]
    # Output shape
    dimensions.output_shape.feature_count = input_shape[0]
    dimensions.output_shape.y = input_shape[1]
    dimensions.output_shape.x = input_shape[2]

    return dimensions


def identity_weights(feature_count: int) -> List[List[List[List[int]]]]:
    """
    identity_weights - Return weights that correspond to identity operation,
                       assuming that feature_count and channel_count are the same.
    :param feature_count:  int  Number of input features
    :return:
        list    Weights for identity operation
    """
    return [
        [[[int(i == j)]] for j in range(feature_count)] for i in range(feature_count)
    ]


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
