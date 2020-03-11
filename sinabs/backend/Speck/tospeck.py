from warnings import warn

import torch
import torch.nn as nn
from sinabs import Network
import sinabs.layers as sl
from typing import Dict, List, Tuple, Union, Optional

# import samna

import speckdemo as sd


SPECK_WEIGHT_PRECISION_BITS = 8
SPECK_STATE_PRECISION_BITS = 16


def to_speck_config(snn: Union[nn.Module, sl.TorchLayer]) -> Dict:
    """
    Build a configuration object of a given module

    :param snn: sinabs.Network or sinabs.layers.TorchLayer instance
    """
    config = sd.configuration.SpeckConfiguration()

    if isinstance(snn, Network):
        layers = list(snn.spiking_model.children())

        i_layer = 0
        i_layer_speck = 0

        # - Iterate over layers from model
        while i_layer < len(layers):

            # Layer to be ported to Speck
            lyr_curr = layers[i_layer]

            if isinstance(lyr_curr, sl.SpikingConv2dLayer):
                # Object representing Speck layer
                speck_layer = config.cnn_layers[i_layer_speck]

                # Extract configuration specs from layer object and update Speck config
                spiking_conv2d_to_speck(lyr_curr, speck_layer)

                # - Consolidate pooling from subsequent layers
                pooling, i_next = consolidate_pooling(layers[i_layer + 1 :])

                # - Destination for CNN layer... make sure that is cnn or sum pooling?

                # For now: Sequential model, second destination always disabled
                speck_layer.destinations[1].enable = False

                if i_next is not None:
                    # Set destination layer
                    speck_layer.destinations[0].layer = i_layer_speck + 1
                    speck_layer.destinations[0].pooling = pooling
                    speck_layer.destinations[0].enable = True

                    # Add 1 to i_layer to go to next layer, + i_next for number
                    # of consolidated pooling layers
                    i_layer += 1 + i_next
                    i_layer_speck += 1

                else:
                    speck_layer.destinations[0].enable = False
                    # TODO: How to route to readout layer? Does destination need to be set?
                    break

            elif isinstance(lyr_curr, sl.SumPooling2dLayer):
                # This case can only happen when `layers` starts with a pooling layer, or
                # input layer because all other pooling layers should get consolidated.
                # Assume that input comes from DVS.

                # Object representing Speck DVS
                dvs = config.dvs_layer
                pooling, i_next = consolidate_pooling(layers[i_layer:], dvs=True)
                dvs.pooling.y, dvs.pooling.x = pooling
                if i_next is not None:
                    dvs.destinations[0].layer = i_layer_speck
                    dvs.destinations[0].enable = True
                    i_layer += i_next
                else:
                    break

            elif isinstance(lyr_curr, sl.InputLayer):
                if i_layer > 0:
                    raise TypeError(
                        f"Only first layer can be of type {type(lyr_curr)}."
                    )
                output_shape = lyr_curr.output_shape
                # - Cut DVS output to match output shape of `lyr_curr`
                dvs = config.dvs_layer
                dvs.cut.y = output_shape[1]
                dvs.cut.x = output_shape[2]
                # TODO: How to deal with feature count?

                i_layer += 1

            else:
                raise TypeError(
                    f"Layers of type {type(lyr_curr)} are currently not supported."
                )

        # TODO: Does anything need to be done after iterating over layers?
        print("Finished configuration of Speck.")

    elif isinstance(snn, sl.TorchLayer):
        # TODO: Do your thing for config
        ...
    elif isinstance(snn, sl.SpikingConv2dLayer):

        # TODO: Next layer??
        destinations = (...,)
        # TODO: which speck layer?
        speck_layer = ...

        spiking_conv2d_to_speck(snn, speck_layer)

    return config


def spiking_conv2d_to_speck(
    layer: sl.SpikingConv2dLayer, speck_layer: sd.configuration.CNNLayerConfig
) -> Dict:

    # Extract configuration specs from layer object
    layer_config = spiking_conv2d_to_dict(layer)
    # Update configuration of the Speck layer
    speck_layer.set_dimensions(**layer_config["dimensions"])
    speck_layer.set_weights(layer_config["weights"])
    speck_layer.set_biases(layer_config["biases"])
    if layer_config["neurons_state"] is not None:
        speck_layer.set_neurons_state(layer_config["neurons_state"])
    for param, value in layer_config["layer_params"].items():
        setattr(speck_layer, param, value)


def consolidate_pooling(
    layers, dvs: bool = False
) -> Tuple[Union[int, Tuple[int], None], int]:
    """
    consolidate_pooling - Consolidate the first `SumPooling2dLayer`s in `layers`
                          until the first object of different type.
    :param layers:  Iterable, containing `SumPooling2dLayer`s and other objects.
    :param dvs:     bool, if True, x- and y- pooling may be different and a
                          Tuple is returned instead of an integer.
    :return:
        int or tuple, consolidated pooling size. Tuple if `dvs` is true.
        int or None, index of first object in `layers` that is not a
                     `SumPooling2dLayer`, or `None`, if all objects in `layers`
                     are `SumPooling2dLayer`s.
    """

    pooling = (1, 1) if dvs else 1

    for i_next, lyr in enumerate(layers):
        if isinstance(lyr, sl.SumPooling2dLayer):
            # Update pooling size
            new_pooling = get_sumpool2d_pooling_size(lyr, dvs=dvs)
            if dvs:
                pooling[0] *= new_pooling[0]
                pooling[1] *= new_pooling[1]
            else:
                pooling *= new_pooling
        else:
            return pooling, i_next

    # If this line is reached, all objects in `layers` are `SumPooling2dLayer`s.
    return pooling, None


def get_sumpool2d_pooling_size(
    layer: sl.SumPooling2dLayer, dvs: bool = True
) -> Union[int, Tuple[int]]:
    """
    get_sumpool2d_pooling_size - Determine the pooling size of a `SumPooling2dLayer` object.
    :param layer:  `SumPooling2dLayer` object
    :param dvs:    bool - If True, pooling does not need to be symmetric.
    :return:
        int or tuple - pooling size. If `dvs` is true, then return a tuple with
                       sizes for y- and x-pooling.
    """
    summary = layer.summary()
    if any(pad != 0 for pad in summary["Padding"]):
        warn(
            f"SumPooling2dLayer `{layer.layer_name}`: Padding is not supported for pooling layers."
        )

    if dvs:
        pooling_y, pooling_x = summary["Pooling"]
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
        return pooling


def spiking_conv2d_to_dict(layer: sl.SpikingConv2dLayer) -> Dict:
    """
    spiking_conv2d_to_dict - Extract a dict with parameters from a `SpikingConv2dLayer`
                             so that they can be written to a Speck configuration.
    :param layer:   SpikingConv2dLayer whose parameters should be extracted
    :return:
        Dict    Parameters of `layer`
    """

    summary = layer.summary()

    # - Layer dimension parameters
    dimensions = dict()

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
    dimensions["padding_x"] = padding_x
    dimensions["padding_y"] = padding_y

    # - Stride
    dimensions["stride_y"], dimensions["stride_x"] = summary["Stride"]

    # - Kernel size
    dimensions["kernel_size"] = summary["Kernel"][0]
    if dimensions["kernel_size"] != summary["Kernel"][1]:
        raise ValueError(
            f"SpikingConv2dLayer `{layer.layer_name}` Kernel must have same height and width."
        )

    # - Input and output shapes
    dimensions["channel_count"] = summary["Input_Shape"][0]
    dimensions["input_size_y"] = summary["Input_Shape"][1]
    dimensions["input_size_x"] = summary["Input_Shape"][2]
    dimensions["output_feature_count"] = summary["Output_Shape"][0]
    dimensions["output_size_y"] = summary["Output_Shape"][1]
    dimensions["output_size_x"] = summary["Output_Shape"][2]

    # - Neuron states
    if layer.state is None:
        neurons_state = None
    else:
        neurons_state = layer.state.transpose(2, 3).tolist()

    # - Resetting vs returning to 0
    return_to_zero = layer.membrane_subtract is not None
    if return_to_zero and layer.membrane_reset != 0:
        warn(
            f"SpikingConv2dLayer `{layer.layer_name}`: Resetting of membrane potential is always to 0."
        )
    elif (not return_to_zero) and layer.membrane_subtract != layer.threshold:
        warn(
            f"SpikingConv2dLayer `{layer.layer_name}`: Subtraction of membrane potential is always by high threshold."
        )

    # - Weights and biases
    if layer.bias:
        weights, biases = layer.parameters()
    else:
        weights, = layer.parameters()
        biases = torch.zeros(layer.channels_out)
    # Transpose last two dimensions of weights to match cortexcontrol
    weights.detach_()
    weights.transpose_(2, 3)

    # - Lower and upper thresholds in a tensor for easier handling
    thresholds = torch.tensor((layer.threshold_low, layer.threshold))

    # - Scaling of weights, biases and thresholds
    # Determine by which common factor weights, biases and thresholds can be scaled
    # such each they matches its precision specificaitons.
    scaling_w = determine_discretization_scale(weights, SPECK_WEIGHT_PRECISION_BITS)
    scaling_b = determine_discretization_scale(biases, SPECK_WEIGHT_PRECISION_BITS)
    scaling_t = determine_discretization_scale(thresholds, SPECK_STATE_PRECISION_BITS)
    scaling = min(scaling_w, scaling_b, scaling_t)
    # Scale weights, biases and thresholds with common scaling factor and discretize
    weights = discretize(weights, scaling)
    biases = discretize(biases, scaling)
    threshold_low, threshold_high = discretize(thresholds, scaling)

    layer_params = dict(
        return_to_zero=return_to_zero,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        monitor_enable=True,  # Yes or no?
        leak_enable=layer.bias,
    )

    return {
        "layer_params": layer_params,
        "dimensions": dimensions,
        "weights": weights.tolist(),
        "biases": biases.tolist(),
        "neurons_state": neurons_state,
    }


def determine_discretization_scale(obj: torch.Tensor, bit_precision: int):
    """
    determine_discretization_scale - Determine how much the values of a torch tensor
                                     can be scaled in order to fit the given precision
    :param obj:            torch.Tensor that is to be scaled
    :param bit_precision:  int - The precision in bits
    :return:
        float   The scaling factor
    """

    # Discrete range
    min_val_disc = 2 ** (bit_precision - 1)
    max_val_disc = 2 ** (bit_precision - 1) - 1

    # Range in which values lie
    min_val_obj = torch.min(obj)
    max_val_obj = torch.max(obj)

    # Determine if negative or positive values are to be considered for scaling
    # Take into account that range for diescrete negative values is slightly larger than for positive
    min_max_ratio_disc = abs(min_val_disc / max_val_disc)
    if abs(min_val_obj) <= abs(max_val_obj) * min_max_ratio_disc:
        scaling = abs(max_val_disc / max_val_obj)
    else:
        scaling = abs(min_val_disc / min_val_obj)

    return scaling


def discretize(obj: torch.Tensor, scaling: float):
    """
    discretize - Scale a torch.Tensor and cast it to discrete integer values
    :param obj:         torch.Tensor that is to be discretized
    :param scaling:     float - Scaling factor to be applied before discretization
    :return:
        torch.Tensor - Scaled and discretized copy of `obj`.
    """

    # Scale the values
    obj_scaled = obj * scaling

    # Round and cast to integers
    obj_scaled_rounded = torch.round(obj_scaled).int()

    return obj_scaled_rounded


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


# def write_to_device(config: Dict, device: samna.SpeckModel, weights=None):
#     """
#     Write your model configuration to dict

#     :param config:
#     :param device:
#     :return:
#     """
#     device.set_config(to_speck_config(config))
#     if weights:
#         device.set_weights(weights)
#     device.apply()


# def to_speck_config(config: Dict) -> samna.SpeckConfig:
#     speck_config = samna.SpeckConfig()
#     # TODO

#     # Populate the config
#     return speck_config
