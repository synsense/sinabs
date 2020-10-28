from .iaf import SpikingLayer
import torch
import torch.nn as nn
import pandas as pd
from typing import Optional, Tuple


class SpikingTemporalConv1dLayer(SpikingLayer):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_shape: int,
        dilation: int = 1,
        strides: int = 1,
        bias: bool = True,
        threshold: float = 1.0,
        threshold_low: Optional[float] = -1.0,
        membrane_subtract: Optional[float] = None,
        membrane_reset: Optional[float] = None,
        layer_name: str = "tc",
        negative_spikes: bool = False
    ):
        """
        Temporal Convolutional Spiking layer. This layer performs wave net like streaming computation,
        where the neuron convolves input at time `t` , `t-1*dilation` .. `t-kernel_shape*dilation` to produce output at time `t`

        :param channels_in: Number of input channels
        :param channels_out: Number of output channels
        :param kernel_shape: Kernel size for temporal convolution
        :param dilation: number of time steps between each kernel step,
        :param stride: Stride length
        :param bias: If this layer has a bias value
        :param threshold: Spiking threshold of the neuron
        :param threshold_low: Lower bound for membrane potential
        :param membrane_subtract: Upon spiking if the membrane potential is subtracted as opposed to reset, what is its value
        :param membrane_reset: What is the reset membrane potential of the neuron
        :param layer_name: Name of this layer

        NOTE: SUBTRACT superseeds Reset value
        """
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.kernel_shape = kernel_shape
        self.bias = bias
        super().__init__(
            (channels_in, None),
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
            layer_name=layer_name,
            negative_spikes=negative_spikes
        )

        self.conv = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=kernel_shape,
            dilation=dilation,
            stride=strides,
            bias=bias,
        )


        # Initialize buffer
        self.len_delay_buffer = ((kernel_shape - 1) * dilation + 1) - 1
        if self.len_delay_buffer:
            self.register_buffer(
                "delay_buffer", torch.zeros((channels_in, self.len_delay_buffer))
            )

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method convolves the input spikes to compute the synaptic input currents to the neuron states

        :param input_spikes: torch.Tensor input to the layer. [Time, Channel]
        :return:  torch.Tensor - synaptic output current [Time, Channel]
        """
        # Convolve all inputs at once
        input_spikes = torch.transpose(input_spikes, 0, 1)
        if self.len_delay_buffer:
            input_spikes = torch.cat((self.delay_buffer, input_spikes), axis=1)
            self.delay_buffer = input_spikes[:, -self.len_delay_buffer :]
        print(input_spikes)
        syn_out = self.conv(input_spikes.unsqueeze(0))  # Add a batch
        syn_out = torch.transpose(syn_out[0], 0, 1)  # Remove the batch and transpose to [Time, Channel]
        return syn_out

    def clear_buffer(self):
        """
        Clear the buffer for this layer ie. zero the delay_buffer
        """
        if self.len_delay_buffer:
            self.delay_buffer.zero_()


    def summary(self) -> pd.Series:
        """
        Returns a summary of the current layer

        :return: pandas Series object
        """
        summary = pd.Series(
            {
                "Type": self.__class__.__name__,
                "Layer": self.layer_name,
                "Output_Shape": self.output_shape,
                "Input_Shape": self.input_shape,
                "Kernel": self.kernel_shape,
                "Fanout_Prev": self.kernel_shape*self.channels_out,
                "Neurons": self.channels_out,
                "Kernel_Params": self.channels_in
                * self.channels_out
                * self.kernel_shape,
                "Bias_Params": self.bias * self.channels_out,
            }
        )
        return summary

    def get_output_shape(self, input_shape) -> Tuple:
        """
        Returns the shape of output, given an input to this layer

        :param input_shape: (channels_in, length)
        :return: (channels_out, length)
        """
        (channels, length) = input_shape
        return self.channels_out, length
