import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from operator import mul
from functools import reduce
from sinabs.layers.quantize import QuantizeLayer
from collections import OrderedDict
from sinabs.cnnutils import compute_padding
from .iaf import SpikingLayer

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingConvTranspose2dLayer(SpikingLayer):
    def __init__(
        self,
        channels_in: int,
        image_shape: ArrayLike,
        channels_out: int,
        kernel_shape: ArrayLike,
        strides: ArrayLike = (1, 1),
        padding: ArrayLike = (0, 0, 0, 0),
        output_padding: ArrayLike = (0, 0),
        bias: bool = True,
        threshold: float = 1.,
        threshold_low: Optional[float] = -1.,
        membrane_subtract: Optional[float] = None,
        membrane_reset: Optional[float] = None,
        layer_name: str = "conv2d",
        negative_spikes: bool = False
    ):
        """
        Pytorch implementation of a spiking iaf neuron which deconvolves 2D inputs, with multiple channels

        :param channels_in: Number of input channels
        :param image_shape: [Height, Width]
        :param channels_out: Number of output channels
        :param kernel_shape: Size of the kernel  (tuple)
        :param strides: Strides in each direction (tuple of size 2)
        :param padding: Padding in each of the 4 directions (left, right, top, bottom)
        :param output_padding: Padding in each of the 4 directions (pad_height, pad_width)
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: Upon spiking, if the membrane potential is subtracted as opposed to reset, \
        what is the subtracted value? Defaults to threshold.
        :param membrane_reset: What is the reset membrane potential of the neuron. \
        If not None, the membrane potential is reset instead of subtracted on spiking.
        :param layer_name: Name of this layer
        """
        SpikingLayer.__init__(
            self,
            input_shape=(channels_in, *image_shape),
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            layer_name=layer_name,
            negative_spikes=negative_spikes
        )
        warnings.warn(
            "SpikingConvTranspose2dLayer deprecated. Use nn.ConvTranspose2d + SpikingLayer instead",
            DeprecationWarning,
            stacklevel=2,
        )

        # Initialize the computational layers
        if padding != (0, 0, 0, 0):
            self.pad = nn.ZeroPad2d(padding)
        else:
            self.pad = None

        self.conv = nn.ConvTranspose2d(
            in_channels=channels_in,
            out_channels=channels_out,
            kernel_size=kernel_shape,
            stride=strides,
            output_padding=output_padding,
            bias=bias,
        )

        # Layer convolutional properties
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.out_padding = output_padding
        self.strides = strides
        self.bias = bias

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method convolves the input spikes to compute the synaptic input currents to the neuron states

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        # Convolve all inputs at once
        if self.pad is None:
            syn_out = self.conv(input_spikes)
        else:
            syn_out = self.conv(self.pad(input_spikes))
        return syn_out

    def summary(self) -> pd.Series:
        """
        Returns a summary of the current layer

        :return: pandas Series object
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Output_Shape": tuple(self.output_shape),
                "Input_Shape": tuple(self.input_shape),
                "Kernel": tuple(self.kernel_shape),
                "Padding": tuple(self.padding),
                "Stride": tuple(self.strides),
                "Fanout_Prev": reduce(
                    mul, np.array(self.kernel_shape) / np.array(self.strides), 1
                )
                * self.channels_out,
                "Neurons": reduce(mul, list(self.output_shape), 1),
                "Kernel_Params": self.channels_in
                * self.channels_out
                * reduce(mul, self.kernel_shape, 1),
                "Bias_Params": self.bias * self.channels_out,
            }
        )
        return summary

    def get_output_shape(self, input_shape) -> Tuple:
        """
        Returns the shape of output, given an input to this layer

        :param input_shape: (channels, height, width)
        :return: (channelsOut, height_out, width_out)
        """
        (channels, height, width) = input_shape

        height_out = (
            (height + sum(self.padding[2:]) - 1) * self.strides[0]
            + self.kernel_shape[0]
            + self.out_padding[0]
        )

        width_out = (
            (width + sum(self.padding[:2]) - 1) * self.strides[1]
            + self.kernel_shape[1]
            + self.out_padding[1]
        )

        return self.channels_out, height_out, width_out


def from_convtranspose2d_keras_conf(
    layer_config: dict,
    input_shape: ArrayLike,
    spiking: bool = False,
    quantize_analog_activation: bool = False,
) -> List:
    """
    Load ConvTranspose2D layer from Json configuration

    :param layer_config: keras configuration dictionary for this object
    :param input_shape: input data shape to determine output dimensions (channels, height, width)
    :param spiking: bool True if spiking layer needs to be loaded
    :param quantize_analog_activation: Whether or not to add a quantization layer for the analog model
    :return: [(layer_name, nn.Module)] Returns a list of layers and their names
    """
    raise NotImplementedError

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
            membrane_subtract=None,
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
