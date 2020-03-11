from typing import Union
from copy import deepcopy

import torch
import torch.nn as nn
import sinabs.layers as sl

SPECK_WEIGHT_PRECISION_BITS = 8
SPECK_STATE_PRECISION_BITS = 16


def discretize_sl(
    snn: Union[nn.Module, sl.TorchLayer]
) -> Union[nn.Module, sl.TorchLayer]:
    """
    discretize - Return a copy of the provided model or layer with discretized,
                 weights, biases, neuron states, and thresholds.
    :param snn:  The model or layer that is to be discretized
    """
    model_copy = deepcopy(snn)
    return discretize_sl_(model_copy)


def discretize_sl_(
    snn: Union[nn.Module, sl.TorchLayer], inplace=False
) -> Union[nn.Module, sl.TorchLayer]:
    """
    discretize_sl_ - Discretize the weights, biases, neuron states, and thresholds
                     of the provided layer's.
    :param snn:  The model or layer that is to be discretized
    """
    if isinstance(snn, sl.SpikingConv2dLayer):
        return _discretize_SC2D_(snn)

    elif isinstance(snn, (sl.InputLayer, sl.SumPooling2dLayer)):
        # - Do not discretize `InputLayer` and `SumPooling2dLayer`
        return snn

    elif isinstance(snn, nn.Module):
        # - For every other type of `Module`s, try discretizing its children
        for lyr in snn.children():
            discretize_sl_(lyr)
        return snn

    else:
        raise ValueError(f"Objects of type `{type(snn)}` are not supported.")


def _discretize_SC2D_(layer: sl.TorchLayer):
    # - Lower and upper thresholds in a tensor for easier handling
    thresholds = torch.tensor((layer.threshold_low, layer.threshold))
    # - Weights and biases
    if layer.bias:
        weights, biases = layer.parameters()
    else:
        weights, = layer.parameters()
        biases = torch.zeros(layer.channels_out)

    # - Scaling of weights, biases, thresholds and neuron states
    # Determine by which common factor weights, biases and thresholds can be scaled
    # such each they matches its precision specificaitons.
    scaling_w = determine_discretization_scale(weights, SPECK_WEIGHT_PRECISION_BITS)
    scaling_b = determine_discretization_scale(biases, SPECK_WEIGHT_PRECISION_BITS)
    scaling_t = determine_discretization_scale(thresholds, SPECK_STATE_PRECISION_BITS)
    if layer.state is not None:
        scaling_n = determine_discretization_scale(
            layer.state, SPECK_STATE_PRECISION_BITS
        )
        scaling = min(scaling_w, scaling_b, scaling_t, scaling_n)
        # Scale neuron state with common scaling factor and discretize
        layer.state = discretize_tensor(layer.state, scaling)
    else:
        scaling = min(scaling_w, scaling_b, scaling_t)
    # Scale weights, biases and thresholds with common scaling factor and discretize
    weights.data = discretize_tensor(weights, scaling)
    biases.data = discretize_tensor(biases, scaling)
    layer.threshold_low, layer.threshold = discretize_tensor(thresholds, scaling)

    return layer


def determine_discretization_scale(obj: torch.Tensor, bit_precision: int) -> float:
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


def discretize_tensor(obj: torch.Tensor, scaling: float) -> torch.Tensor:
    """
    discretize_tensor - Scale a torch.Tensor and cast it to discrete integer values
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
