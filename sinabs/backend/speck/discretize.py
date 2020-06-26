from typing import Optional
from copy import deepcopy

import torch
import torch.nn as nn
import sinabs.layers as sl

SPECK_WEIGHT_PRECISION_BITS = 8
SPECK_STATE_PRECISION_BITS = 16


def discretize_conv_spike(
    conv_lyr: nn.Conv2d, spike_lyr: sl.SpikingLayerBPTT, to_int: bool = True,
) -> (nn.Conv2d, sl.SpikingLayerBPTT):
    """Discretize convolutional and spiking layers together.

    This function takes a 2D convolutional and a spiking layer and returns a
    copy of each, with discretized weights, bias and threshold.

    Parameters
    ----------
    conv_lyr: nn.Conv2d
        Convolutional layer
    spike_lyr: sl.SpikingLayerBPTT
        Spiking layer
    to_int: bool
        Use integer types for discretized parameter

    Returns
    -------
    nn.Conv2d
        Discretized copy of convolutional layer
    sl.SpikingLayerBPTT
        Discretized copy of spiking layer

    """
    conv_lyr_copy = deepcopy(conv_lyr)
    spike_lyr_copy = deepcopy(spike_lyr)
    return discretize_conv_spike_(conv_lyr_copy, spike_lyr_copy, to_int=to_int)


def discretize_conv_spike_(
    conv_lyr: nn.Conv2d, spike_lyr: sl.SpikingLayerBPTT, to_int: bool = True,
) -> (nn.Conv2d, sl.SpikingLayerBPTT):
    """Discretize convolutional and spiking layers together, in-place.

    This function takes a 2D convolutional and a spiking layer and discretizes
    weights, bias and threshold in-place.

    Parameters
    ----------
    conv_lyr: nn.Conv2d
        Convolutional layer
    spike_lyr: sl.SpikingLayerBPTT
        Spiking layer
    to_int: bool
        Use integer types for discretized parameter

    Returns
    -------
    nn.Conv2d
        Discretized convolutional layer
    sl.SpikingLayerBPTT
        Discretized spiking layer

    """

    return _discretize_conv_spk_(conv_lyr, spike_lyr, to_int=to_int)


def discretize_conv(
    layer: nn.Conv2d,
    spk_thr: float,
    spk_thr_low: float,
    spk_state: Optional[torch.Tensor] = None,
    to_int: bool = True,
):
    """Discretize convolutional layer.

    This function takes a 2D convolutional layer and parameters of a subsequent
    spiking layer to return a discretized copy of the convolutional layer.

    Parameters
    ----------
    layer: nn.Conv2d
        Convolutional layer
    spk_thr: float
        Upper threshold of subsequent spiking layer
    spk_thr_low: float
        Lower threshold of subsequent spiking layer
    to_int: bool
        Use integer types for discretized parameter

    Returns
    -------
    nn.Conv2d
        Discretized copy of convolutional layer

    """
    lyr_copy = deepcopy(layer)
    layer_discr = discretize_conv_(
        layer=lyr_copy,
        spk_thr=spk_thr,
        spk_thr_low=spk_thr_low,
        spk_state=spk_state,
        to_int=to_int,
    )
    return layer_discr


def discretize_conv_(
    layer: nn.Conv2d,
    spk_thr: float,
    spk_thr_low: float,
    spk_state: Optional[torch.Tensor] = None,
    to_int: bool = True,
):
    """Discretize convolutional layer, in-place.

    This function discretizes a 2D convolutional layer in-place, based on
    parameters of a subsequent spiking layer.

    Parameters
    ----------
    layer: nn.Conv2d
        Convolutional layer
    spk_thr: float
        Upper threshold of subsequent spiking layer
    spk_thr_low: float
        Lower threshold of subsequent spiking layer
    to_int: bool
        Use integer types for discretized parameter

    Returns
    -------
    nn.Conv2d
        Discretized convolutional layer

    """
    layer_discr, __ = _discretize_conv_spk_(
        conv_lyr=layer,
        spk_thr=spk_thr,
        spk_thr_low=spk_thr_low,
        spk_state=spk_state,
        to_int=to_int,
    )
    return layer_discr


def discretize_spk(
    layer: sl.SpikingLayerBPTT,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    to_int: bool = True,
):
    """Discretize spiking layer.

    This function takes a spiking layer and parameters of a preceding
    convolutional layer to return a discretized copy of the spiking layer.

    Parameters
    ----------
    layer: sl.SpikingLayerBPTT
        Spiking layer
    conv_weight: torch.Tensor
        Weight tensor of preceding convolutional layer
    conv_bias: torch.Tensor or None
        Bias of preceding convolutional layer
    to_int: bool
        Use integer types for discretized parameter

    Returns
    -------
    sl.SpikingLayerBPTT
        Discretized copy of spiking layer

    """
    lyr_copy = deepcopy(layer)
    layer_discr = discretize_conv_(
        layer=lyr_copy, conv_weight=conv_weight, conv_bias=conv_bias, to_int=to_int
    )
    return layer_discr


def discretize_spk_(
    layer: sl.SpikingLayerBPTT,
    conv_weight: torch.Tensor,
    conv_bias: Optional[torch.Tensor] = None,
    to_int: bool = True,
):
    """Discretize spiking layer in-place.

    This function discretizes a spiking layer in-place, based on parameters of a
    preceding convolutional layer.

    Parameters
    ----------
    layer: sl.SpikingLayerBPTT
        Spiking layer
    conv_weight: torch.Tensor
        Weight tensor of preceding convolutional layer
    conv_bias: torch.Tensor or None
        Bias of preceding convolutional layer
    to_int: bool
        Use integer types for discretized parameter

    Returns
    -------
    sl.SpikingLayerBPTT
        Discretized spiking

    """
    __, layer_discr = _discretize_conv_spk_(
        conv_lyr=layer, conv_weight=conv_weight, conv_bias=conv_bias, to_int=to_int
    )
    return layer_discr


def _discretize_conv_spk_(
    conv_lyr: Optional[nn.Conv2d] = None,
    spike_lyr: Optional[sl.SpikingLayerBPTT] = None,
    spk_thr: Optional[float] = None,
    spk_thr_low: Optional[float] = None,
    spk_state: Optional[torch.Tensor] = None,
    conv_weight: Optional[torch.Tensor] = None,
    conv_bias: Optional[torch.Tensor] = None,
    to_int: bool = True,
):
    """Discretize convolutional and spiking layer

    Determine and apply a suitable scaling factor for weight and bias of
    convolutional layer as well as thresholds and state of spiking layer, taking
    into account current parameters and available precision on Speck. Instead of
    providing layers, respective parameters can be provided directly. If a layer
    is not provided, `None` will be returned instead of its discrete version.

    Parameters
    ----------
        conv_lyr: nn.Conv2d or None
            Convolutional layer
        spike_lyr: sl.SpikingLayerBPTT or None
            Spiking layer
        spk_thr: float or None
            Upper threshold of spiking layer. Has to be provided if `spike_lyr` is `None`.
            Is ignored otherwise.
        spk_thr_low: float or None
            Lower threshold of spiking layer. Has to be provided if `spike_lyr` is `None`.
            Is ignored otherwise.
        spk_state: torch.Tensor or None
            State of spiking layer. Igrnoed if `spike_lyr` is not `None`.
        conv_weight: torch.Tensor or None
            Weight of convolutional layer. Has to be provided if `conv_lyr` is `None`.
            Is ignored otherwise.
        conv_bias: torch.Tensor or None
            Bias of convolutional layer. Ignored if `conv_lyr` is not `None`.
        to_int: bool
            Use integer types for discretized parameters.

    Returns
    -------
        nn.Conv2d or None
            Discretized convolutional layer if `conv_lyr` is not `None`, else `None`
        sl.SpikingLayerBPTT or None
            Discretized spiking layer if `spk_lyr` is not `None`, else `None`
    """

    if conv_lyr is None:
        discr_conv = False

        if conv_weight is None:
            raise TypeError("If `conv_lyr` is `None`, `wegiht` must be provided.")

        if conv_bias is None:
            conv_bias = torch.zeros(conv_weight.shape[0])

    else:
        if not isinstance(conv_lyr, nn.Conv2d):
            raise TypeError("`conv_lyr` must be of type `Conv2d`")

        discr_conv = True

        # - Weights and bias
        if conv_lyr.bias is not None:
            conv_weight, conv_bias = conv_lyr.parameters()
        else:
            (conv_weight,) = conv_lyr.parameters()
            conv_bias = torch.zeros(conv_lyr.out_channels)

    if spike_lyr is None:

        discr_spk = False

        if spk_thr is None or spk_thr_low is None:
            raise TypeError(
                "If `spk_lyr` is `None`, both `spk_thr` and `spk_thr_low` must be provided."
            )
        # - Lower and upper thresholds in a tensor for easier handling
        thresholds = torch.tensor((spk_thr_low, spk_thr))
    else:
        if not isinstance(spike_lyr, sl.SpikingLayerBPTT):
            raise TypeError("`spike_lyr` must be of type `SpikingLayer`")

        discr_spk = True

        # - Lower and upper thresholds in a tensor for easier handling
        thresholds = torch.tensor((spike_lyr.threshold_low, spike_lyr.threshold))

        spk_state = spike_lyr.state

    # - Scaling of conv_weight, conv_bias, thresholds and neuron states
    # Determine by which common factor conv_weight, conv_bias and thresholds can be scaled
    # such each they matches its precision specificaitons.
    scaling_w = determine_discretization_scale(conv_weight, SPECK_WEIGHT_PRECISION_BITS)
    scaling_b = determine_discretization_scale(conv_bias, SPECK_WEIGHT_PRECISION_BITS)
    scaling_t = determine_discretization_scale(thresholds, SPECK_STATE_PRECISION_BITS)
    if spk_state is not None:
        scaling_n = determine_discretization_scale(
            spk_state, SPECK_STATE_PRECISION_BITS
        )
        scaling = min(scaling_w, scaling_b, scaling_t, scaling_n)
        # Scale neuron state with common scaling factor and discretize
        spk_state = discretize_tensor(spk_state, scaling, to_int=to_int)
    else:
        scaling = min(scaling_w, scaling_b, scaling_t)

    # Scale conv_weight, conv_bias and thresholds with common scaling factor and discretize
    if discr_conv:
        conv_weight.data = discretize_tensor(conv_weight, scaling, to_int=to_int)
        conv_bias.data = discretize_tensor(conv_bias, scaling, to_int=to_int)
    if discr_spk:
        spike_lyr.threshold_low, spike_lyr.threshold = (
            discretize_tensor(thresholds, scaling, to_int=to_int).detach().numpy()
        )
        spike_lyr.membrane_subtract = discretize_scalar(
            spike_lyr.membrane_subtract, scaling
        )

    return conv_lyr, spike_lyr


def determine_discretization_scale(obj: torch.Tensor, bit_precision: int) -> float:
    """Determine a scale for discretization

    Determine how much the values of a torch tensor can be scaled in order to fit
    the given precision

    Parameters
    ----------
        obj: torch.Tensor
            Tensor that is to be scaled
        bit_precision: int
            The precision in bits

    Returns
    -------
        float
            The scaling factor
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


def discretize_tensor(
    obj: torch.Tensor, scaling: float, to_int: bool = True
) -> torch.Tensor:
    """Scale a torch.Tensor and cast it to discrete integer values

    Parameters
    ----------
        obj: torch.Tensor
            Tensor that is to be discretized
        scaling: float
            Scaling factor to be applied before discretization
        to_int: bool
            If False, round the values, but don't cast to Int. (Default True).

    Returns
    -------
        torch.Tensor
            Scaled and discretized copy of `obj`.
    """

    # Scale the values
    obj_scaled = obj * scaling

    # Round and cast to integers
    obj_scaled_rounded = torch.round(obj_scaled)

    if to_int:
        obj_scaled_rounded = obj_scaled_rounded.int()

    return obj_scaled_rounded


def discretize_scalar(obj: float, scaling: float) -> int:
    """Scale a float and cast it to discrete integer values

    Parameters
    ----------
        obj: float
            Value that is to be discretized
        scaling: float
            Scaling factor to be applied before discretization

    Returns
    -------
        int
            Scaled and discretized copy of `obj`.
    """

    # Scale the values
    obj_scaled = obj * scaling

    # Round and cast to integers
    return int(obj_scaled)


### OBSOLETE

# def discretize_sl(
#     snn: Union[nn.Module, sl.TorchLayer], to_int: bool = True
# ) -> Union[nn.Module, sl.TorchLayer]:
#     """
#     discretize - Return a copy of the provided model or layer with discretized,
#                  weights, biases, neuron states, and thresholds.
#     :param snn:  The model or layer that is to be discretized
#     :param to_int: If False, round the values, but don't cast to Int. (Default True).
#     """
#     try:
#         model_copy = deepcopy(snn)
#     except RuntimeError:
#         raise NotImplementedError(
#             "Some sinabs object can currently not be copied. You may run "
#             "`discretize_sl_` instead, to discretize the original model instead "
#             "of a copy."
#         )
#     return discretize_sl_(model_copy, to_int=to_int)


# def discretize_sl_(
#     snn: Union[nn.Module, sl.TorchLayer], to_int: bool = True
# ) -> Union[nn.Module, sl.TorchLayer]:
#     """
#     discretize_sl_ - Discretize the weights, biases, neuron states, and thresholds
#                      of the provided layers.
#     :param snn:  The model or layer that is to be discretized
#     :param to_int: If False, round the values, but don't cast to Int. (Default True).
#     """
#     if isinstance(snn, sl.SpikingConv2dLayer):
#         return _discretize_SC2D_(snn, to_int=to_int)

#     elif isinstance(snn, (sl.InputLayer, sl.SumPooling2dLayer)):
#         # - Do not discretize `InputLayer` and `SumPooling2dLayer`
#         return snn

#     elif isinstance(snn, nn.Module):
#         # - For every other type of `Module`s, try discretizing its children
#         for lyr in snn.children():
#             discretize_sl_(lyr, to_int=to_int)
#         return snn

#     else:
#         raise ValueError(f"Objects of type `{type(snn)}` are not supported.")


# def _discretize_SC2D_(layer: sl.TorchLayer, to_int: bool):
#     # - Lower and upper thresholds in a tensor for easier handling
#     thresholds = torch.tensor((layer.threshold_low, layer.threshold))
#     # - Weights and biases
#     if layer.bias:
#         weights, biases = layer.parameters()
#     else:
#         (weights,) = layer.parameters()
#         biases = torch.zeros(layer.channels_out)

#     # - Scaling of weights, biases, thresholds and neuron states
#     # Determine by which common factor weights, biases and thresholds can be scaled
#     # such each they matches its precision specificaitons.
#     scaling_w = determine_discretization_scale(weights, SPECK_WEIGHT_PRECISION_BITS)
#     scaling_b = determine_discretization_scale(biases, SPECK_WEIGHT_PRECISION_BITS)
#     scaling_t = determine_discretization_scale(thresholds, SPECK_STATE_PRECISION_BITS)
#     if layer.state is not None:
#         scaling_n = determine_discretization_scale(
#             layer.state, SPECK_STATE_PRECISION_BITS
#         )
#         scaling = min(scaling_w, scaling_b, scaling_t, scaling_n)
#         # Scale neuron state with common scaling factor and discretize
#         layer.state = discretize_tensor(layer.state, scaling, to_int=to_int)
#     else:
#         scaling = min(scaling_w, scaling_b, scaling_t)

#     # Scale weights, biases and thresholds with common scaling factor and discretize
#     weights.data = discretize_tensor(weights, scaling, to_int=to_int)
#     biases.data = discretize_tensor(biases, scaling, to_int=to_int)
#     layer.threshold_low, layer.threshold = (
#         discretize_tensor(thresholds, scaling, to_int=to_int).detach().numpy()
#     )
#     layer.membrane_subtract = discretize_scalar(layer.membrane_subtract, scaling)
#     layer.membrane_reset = discretize_scalar(layer.membrane_reset, scaling)

#     return layer
