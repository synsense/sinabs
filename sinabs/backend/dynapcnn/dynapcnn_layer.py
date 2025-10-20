from copy import deepcopy
from functools import partial
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

import sinabs.layers as sl

from .discretize import discretize_conv_spike_

# Define sum pooling functional as power-average pooling with power 1
sum_pool2d = partial(nn.functional.lp_pool2d, norm_type=1)


def convert_linear_to_conv(
    lin: nn.Linear, input_shape: Tuple[int, int, int]
) -> nn.Conv2d:
    """Convert Linear layer to Conv2d.

    Args:
        lin (nn.Linear): linear layer to be converted.
        input_shape (tuple): the tensor shape the layer expects.

    Returns:
        convolutional layer equivalent to `lin`.
    """
    in_chan, in_h, in_w = input_shape
    if lin.in_features != in_chan * in_h * in_w:
        raise ValueError(
            "Shape of linear layer weight does not match provided input shape"
        )

    layer = nn.Conv2d(
        in_channels=in_chan,
        kernel_size=(in_h, in_w),
        out_channels=lin.out_features,
        padding=0,
        bias=lin.bias is not None,
    )

    if lin.bias is not None:
        layer.bias.data = lin.bias.data.clone().detach()

    layer.weight.data = (
        lin.weight.data.clone()
        .detach()
        .reshape((lin.out_features, in_chan, in_h, in_w))
    )

    return layer


class DynapcnnLayer(nn.Module):
    """Create a DynapcnnLayer object representing a layer on DynapCNN or Speck.

    Requires a convolutional layer, a sinabs spiking layer and a list of
    pooling values. The layers are used in the order conv -> spike -> pool.

    Attributes:
        conv: torch.nn.Conv2d or torch.nn.Linear. Convolutional or linear layer.
            Linear will be converted to convolutional.
        spk (sinabs.layers.IAFSqueeze): Sinabs IAF layer.
        in_shape (tuple of int):  The input shape, needed to create dynapcnn configs
            if the network does not contain an input layer.
            Convention: (features, height, width).
        pool (List of integers): Each integer entry represents an output (destination
            on chip) and whether pooling should be applied (values > 1) or not
            (values equal to 1). The number of entries determines the number of tensors
            the layer's forward method returns.
        discretize (bool): Whether to discretize parameters.
        rescale_weights (int): Layer weights will be multiplied by this value.
    """

    def __init__(
        self,
        conv: nn.Conv2d,
        spk: sl.IAFSqueeze,
        in_shape: Tuple[int, int, int],
        pool: List[int],
        discretize: bool = True,
        rescale_weights: int = 1,
    ):
        super().__init__()

        self.in_shape = in_shape
        self.pool = pool
        self._discretize = discretize
        self._rescale_weights = rescale_weights

        if not isinstance(spk, sl.IAFSqueeze):
            raise TypeError(
                f"Unsupported spiking layer type {type(spk)}. "
                "Only `IAFSqueeze` layers are supported."
            )
        spk = deepcopy(spk)

        # Convert `nn.Linear` to `nn.Conv2d`.
        if isinstance(conv, nn.Linear):
            conv = convert_linear_to_conv(conv, in_shape)
            if spk.is_state_initialised() and (ndim := spk.v_mem.ndim) < 4:
                for __ in range(4 - ndim):
                    # Expand spatial dimensions
                    spk.v_mem = spk.v_mem.data.unsqueeze(-1)
        else:
            conv = deepcopy(conv)

        if self._rescale_weights != 1:
            # this has to be done after copying but before discretizing
            conv.weight.data = (conv.weight * self._rescale_weights).clone().detach()

        # check if convolution kernel is a square.
        if conv.kernel_size[0] != conv.kernel_size[1]:
            raise ValueError(
                "The kernel of a `nn.Conv2d` must have the same height and width."
            )
        for pool_size in pool:
            if pool_size[0] != pool_size[1]:
                raise ValueError("Only square pooling kernels are supported")

        # int conversion is done while writing the config.
        if self._discretize:
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        self.conv = conv
        self.spk = spk

    @property
    def conv_layer(self):
        return self.conv

    @property
    def spk_layer(self):
        return self.spk

    @property
    def discretize(self):
        return self._discretize

    @property
    def rescale_weights(self):
        return self._rescale_weights

    @property
    def conv_out_shape(self):
        return self._get_conv_output_shape()

    def forward(self, x) -> List[torch.Tensor]:
        """Torch forward pass.

        ...
        """

        returns = []

        x = self.conv_layer(x)
        x = self.spk_layer(x)

        for pool in self.pool:
            if pool == 1:
                # no pooling is applied.
                returns.append(x)
            else:
                # sum pooling of `(pool, pool)` is applied.
                pool_out = sum_pool2d(x, kernel_size=pool)
                returns.append(pool_out)

        if len(returns) == 1:
            return returns[0]
        else:
            return tuple(returns)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Call `zero_grad` method of spiking layer"""
        return self.spk.zero_grad(set_to_none)

    def get_neuron_shape(self) -> Tuple[int, int, int]:
        """Return the output shape of the neuron layer.

        Returns:
            conv_out_shape (tuple): formatted as (features, height, width).
        """
        # same as the convolution's output.
        return self._get_conv_output_shape()

    def get_output_shape(self) -> List[Tuple[int, int, int]]:
        """Return the output shapes of the layer, including pooling.

        Returns:
            One entry per destination, each formatted as (features, height, width).
        """
        neuron_shape = self.get_neuron_shape()
        # this is the actual output shape, including pooling
        output_shape = []
        for pool in self.pool:
            output_shape.append(
                neuron_shape[0],
                neuron_shape[1] // pool,
                neuron_shape[2] // pool,
            )
        return output_shape

    def summary(self) -> dict:
        """Returns a summary of the convolution's/pooling's kernel sizes and the output shape of the spiking layer."""

        return {
            "pool": (self.pool),
            "kernel": list(self.conv_layer.weight.data.shape),
            "neuron": self._get_conv_output_shape(),  # neuron layer output has the same shape as the convolution layer ouput.
        }

    def memory_summary(self):
        """Computes the amount of memory required for each of the components. Note that this is not
        necessarily the same as the number of parameters due to some architecture design
        constraints.

        .. math::

            K_{MT} = c \\cdot 2^{\\lceil \\log_2\\left(k_xk_y\\right) \\rceil + \\lceil \\log_2\\left(f\\right) \\rceil}

        .. math::

            N_{MT} = f \\cdot 2^{ \\lceil \\log_2\\left(f_y\\right) \\rceil + \\lceil \\log_2\\left(f_x\\right) \\rceil }

        Returns:
            A dictionary with keys kernel, neuron and bias and the corresponding memory sizes
        """
        summary = self.summary()
        f, c, h, w = summary["kernel"]
        (
            f,
            neuron_height,
            neuron_width,
        ) = (
            self._get_conv_output_shape()
        )  # neuron layer output has the same shape as the convolution layer ouput.

        return {
            "kernel": c * pow(2, np.ceil(np.log2(h * w)) + np.ceil(np.log2(f))),
            "neuron": f
            * pow(2, np.ceil(np.log2(neuron_height)) + np.ceil(np.log2(neuron_width))),
            "bias": 0 if self.conv_layer.bias is None else len(self.conv_layer.bias),
        }

    def _get_conv_output_shape(self) -> Tuple[int, int, int]:
        """Computes the output dimensions of `conv_layer`.

        Returns:
            output dimensions (tuple): a tuple describing `(output channels, height, width)`.
        """
        # get the layer's parameters.

        out_channels = self.conv_layer.out_channels
        kernel_size = self.conv_layer.kernel_size
        stride = self.conv_layer.stride
        padding = self.conv_layer.padding
        dilation = self.conv_layer.dilation

        # compute the output height and width.
        out_height = (
            (self.in_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
            // stride[0]
        ) + 1
        out_width = (
            (self.in_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
            // stride[1]
        ) + 1

        return (out_channels, out_height, out_width)
