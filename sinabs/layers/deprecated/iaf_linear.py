import torch
import warnings
import torch.nn as nn
import numpy as np
import pandas as pd
from .iaf import SpikingLayer
from typing import Optional, Union, List, Tuple


# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]


class SpikingLinearLayer(SpikingLayer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset: Optional[float] = None,
        layer_name: str = "conv1d",
        negative_spikes: bool = False,
    ):
        """
        Spiking Linear/Densely connected layer

        :param in_features: Number of input channels
        :param out_features: Number of output channels
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lower bound for membrane potential
        :param membrane_subtract: Upon spiking, if the membrane potential is subtracted as opposed to reset, \
        what is the subtracted value? Defaults to threshold.
        :param membrane_reset: What is the reset membrane potential of the neuron. \
        If not None, the membrane potential is reset instead of subtracted on spiking.
        :param layer_name: Name of this layer
        """
        SpikingLayer.__init__(
            self,
            input_shape=(in_features,),
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            layer_name=layer_name,
            negative_spikes=negative_spikes,
        )
        warnings.warn(
            "SpikingLinearLayer deprecated. Use nn.Linear + SpikingLayer instead",
            DeprecationWarning,
            stacklevel=2,
        )

        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Layer convolutional properties
        self.channels_in = in_features
        self.channels_out = out_features
        self.bias = bias

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method convolves the input spikes to compute the synaptic input currents to the neuron states

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """

        return self.linear(input_spikes)

    def summary(self) -> pd.Series:
        """
        Returns a summary of the current layer

        :return: pandas Series object
        """
        bias = 0.0 if self.bias is None else self.bias
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Output_Shape": self.output_shape,
                "Input_Shape": self.input_shape,
                "Fanout_Prev": self.channels_out,
                "Neurons": self.channels_out,
                "Kernel_Params": self.channels_in * self.channels_out,
                "Bias_Params": bias * self.channels_out,
            }
        )
        return summary

    def get_output_shape(self, input_shape) -> Tuple:
        """
        Returns the shape of output, given an input to this layer

        :param input_shape: (in_features,)
        :return: (out_features, )
        """
        return (self.channels_out,)
