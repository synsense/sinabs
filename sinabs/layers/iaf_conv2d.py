##
# iaf_conv2d.py - Torch implementation of a spiking 2D convolutional layer
##

import numpy as np
import pandas as pd
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Dict
from operator import mul
from functools import reduce
from .layer import TorchLayer
from .quantize import QuantizeLayer
from collections import OrderedDict
from sinabs.cnnutils import conv_output_size, compute_padding

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingConv2dLayer(TorchLayer):
    def __init__(
        self,
        channels_in: int,
        image_shape: ArrayLike,
        channels_out: int,
        kernel_shape: ArrayLike,
        strides: ArrayLike = (1, 1),
        padding: ArrayLike = (0, 0, 0, 0),
        bias: bool = True,
        threshold: float = 8,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        membrane_reset: float = 0,
        layer_name: str = "conv2d",
    ):
        """
        Pytorch implementation of a spiking neuron which convolve 2D inputs, with multiple channels

        :param channels_in: Number of input channels
        :param image_shape: [Height, Width]
        :param channels_out: Number of output channels
        :param kernel_shape: Size of the kernel  (tuple)
        :param strides: Strides in each direction (tuple of size 2)
        :param padding: Padding in each of the 4 directions (left, right, top, bottom)
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lowerbound for membrane potential
        :param membrane_subtract: Upon spiking if the membrane potential is subtracted as opposed to reset, what is its value
        :param membrane_reset: What is the reset membrane potential of the neuron
        :param layer_name: Name of this layer

        NOTE: SUBTRACT superseeds Reset value
        """
        TorchLayer.__init__(
            self, input_shape=(channels_in, *image_shape), layer_name=layer_name
        )
        if padding != (0, 0, 0, 0):
            self.pad = nn.ZeroPad2d(padding)
        else:
            self.pad = None
        self.conv = nn.Conv2d(
            channels_in,
            channels_out,
            kernel_size=kernel_shape,
            stride=strides,
            bias=bias,
        )
        # Initialize neuron states
        self.membrane_subtract = membrane_subtract
        self.membrane_reset = membrane_reset
        self.threshold = threshold
        self.threshold_low = threshold_low

        # Layer convolutional properties
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.strides = strides
        self.bias = bias

        # Blank parameter place holders
        self.spikes_number = None
        self.state = None

    @property
    def threshold_low(self):
        return self._threshold_low

    @threshold_low.setter
    def threshold_low(self, new_threshold_low):
        self._threshold_low = new_threshold_low
        if new_threshold_low is None:
            try:
                del self.thresh_lower
            except AttributeError:
                pass
        else:
            # Relu on the layer
            self.thresh_lower = nn.Threshold(new_threshold_low, new_threshold_low)

    def reset_states(self):
        """
        Reset the state of all neurons in this layer
        """
        if self.state is None:
            return
        else:
            self.state.zero_()

    def forward(self, binary_input):
        # Determine no. of time steps from input
        time_steps = len(binary_input)

        # Convolve all inputs at once
        if self.pad is None:
            tsrConvOut = self.conv(binary_input)
        else:
            tsrConvOut = self.conv(self.pad(binary_input))

        # Local variables
        membrane_subtract = self.membrane_subtract
        threshold = self.threshold
        threshold_low = self.threshold_low
        membrane_reset = self.membrane_reset

        # Initialize state as required
        # Create a vector to hold all output spikes
        if self.spikes_number is None or len(self.spikes_number) != len(binary_input):
            del self.spikes_number  # Free memory just to be sure
            self.spikes_number = tsrConvOut.new_zeros(
                time_steps, *tsrConvOut.shape[1:]
            ).int()

        self.spikes_number.zero_()
        spikes_number = self.spikes_number

        if self.state is None:
            self.state = tsrConvOut.new_zeros(tsrConvOut.shape[1:])

        state = self.state

        # Loop over time steps
        for iCurrentTimeStep in range(time_steps):
            state = state + tsrConvOut[iCurrentTimeStep]

            # - Reset or subtract from membrane state after spikes
            if membrane_subtract is not None:
                # Calculate number of spikes to be generated
                spikes_number[iCurrentTimeStep] = (state >= threshold).int() + (
                    state - threshold > 0
                ).int() * ((state - threshold) / membrane_subtract).int()
                # - Subtract from states
                state = state - (
                    membrane_subtract * spikes_number[iCurrentTimeStep].float()
                )
            else:
                # - Check threshold crossings for spikes
                vbRecSpikeRaster = state >= threshold
                # - Add to spike counter
                spikes_number[iCurrentTimeStep] = vbRecSpikeRaster
                # - Reset neuron states
                state = (
                    vbRecSpikeRaster.float() * membrane_reset
                    + state * (vbRecSpikeRaster ^ 1).float()
                )

            if threshold_low is not None:
                state = self.thresh_lower(state)  # Lower bound on the activation

        self.state = state
        self.spikes_number = spikes_number
        return spikes_number.float()  # Float is just to keep things compatible

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

        height_out = conv_output_size(
            height + sum(self.padding[2:]), self.kernel_shape[0], self.strides[0]
        )
        width_out = conv_output_size(
            width + sum(self.padding[:2]), self.kernel_shape[1], self.strides[1]
        )
        return self.channels_out, height_out, width_out


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
        torch_layerPad = nn.ZeroPad2d(padding)
        layer_list.append((layer_name + "_padding", torch_layerPad))
        # Create a convolutional layer
        torch_analogue_layer = nn.Conv2d(
            channels,
            layer_config["config"]["filters"],
            kernel_shape,
            stride=vStride,
            bias=layer_config["config"]["use_bias"],
        )
        layer_list.append((layer_name + "_conv", torch_analogue_layer))
        # Activation
        if quantize_analog_activation:
            layer_nameActivation = (
                layer_name + "_" + layer_config["config"]["activation"]
            )
        else:
            layer_nameActivation = layer_name

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
