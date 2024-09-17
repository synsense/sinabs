# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from copy import deepcopy
from functools import partial
from typing import Tuple, List

import numpy as np
import torch
from torch import nn

import sinabs.layers as sl

from .discretize import discretize_conv_spike_


# Define sum pooling functional as power-average pooling with power 1
sum_pool2d = partial(nn.functional.lp_pool2d, norm_type=1)


class DynapcnnLayer(nn.Module):
    """Create a DynapcnnLayer object representing a layer on DynapCNN or Speck.

    Requires a convolutional layer, a sinabs spiking layer and a list of
    pooling values. The layers are used in the order conv -> spike -> pool.

    Parameters
    ----------
        conv: torch.nn.Conv2d or torch.nn.Linear
            Convolutional or linear layer
            (linear will be converted to convolutional)
        spk: sinabs.layers.IAFSqueeze
            Sinabs IAF layer
        in_shape: tuple of int
            The input shape, needed to create dynapcnn configs if the network
            does not contain an input layer. Convention: (features, height, width)
        pool: List of integers
            Each integer entry represents an output (destination on chip) and
            whether pooling should be applied (values > 1) or not (values equal
            to 1). The number of entries determines the number of tensors the
            layer's forward method returns.
        discretize: bool
            Whether to discretize parameters.
        rescale_weights: int
            Layer weights will be divided by this value.
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

        self.in_shape           = in_shape
        self._pool              = pool
        self._discretize        = discretize
        self._rescale_weights    = rescale_weights

        spk = deepcopy(spk)

        if isinstance(conv, nn.Linear):
            conv = self._convert_linear_to_conv(conv)
            if spk.is_state_initialised():
                # Expand dims
                spk.v_mem = spk.v_mem.data.unsqueeze(-1).unsqueeze(-1)
        else:
            conv = deepcopy(conv)

        if self._rescale_weights != 1:
            # this has to be done after copying but before discretizing
            conv.weight.data = (conv.weight / self._rescale_weights).clone().detach()

        # int conversion is done while writing the config.
        if self._discretize:
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        self._conv               = conv
        self._spk                = spk
    
    @property
    def conv(self):
        return self._conv

    @property
    def spk(self):
        return self._spk

    @property
    def pool(self):
        return self._pool

    @property
    def discretize(self):
        return self._discretize

    @property
    def rescale_weights(self):
        return self._rescale_weights
    
    @property
    def conv_out_shape(self):
        return self._get_conv_output_shape()
        
    ####################################################### Public Methods #######################################################
    
    def forward(self, x) -> List[torch.Tensor]:
        """Torch forward pass.

        ...
        """

        returns = []
        
        x = self.conv(x)
        x = self.spk(x)

        for pool in self._pool:

            if pool == 1:
                # no pooling is applied.
                returns.append(x)
            else:
                # sum pooling of `(pool, pool)` is applied.
                pool_out = sum_pool2d(x, kernel_size=pool)
                returns.append(pool_out)

        return tuple(returns)

    def zero_grad(self, set_to_none: bool = False) -> None:
        """ Call `zero_grad` method of spiking layer """
        return self._spk.zero_grad(set_to_none)
    
    def get_neuron_shape(self) -> Tuple[int, int, int]:
        """Return the output shape of the neuron layer.

        Returns
        -------
        - conv_out_shape (tuple): formatted as (features, height, width).
        """
        # same as the convolution's output.
        return self._get_conv_output_shape()

    def get_output_shape(self) -> List[Tuple[int, int, int]]:
        """Return the output shapes of the layer, including pooling.

        Returns
        -------
        - output_shape (list of tuples): 
            One entry per destination, each formatted as (features, height, width).
        """
        neuron_shape = self.get_neuron_shape()
        # this is the actual output shape, including pooling
        output_shape = []
        for pool in self._pool:
            output_shape.append(
                neuron_shape[0],
                neuron_shape[1] // pool,
                neuron_shape[2] // pool,
            )
        return output_shape

    def summary(self) -> dict:
        """ Returns a summary of the convolution's/pooling's kernel sizes and the output shape of the spiking layer."""

        return {
            "pool": (self._pool),
            "kernel": list(self.conv_layer.weight.data.shape),
            "neuron": self._get_conv_output_shape(),                          # neuron layer output has the same shape as the convolution layer ouput.
        }
    
    def memory_summary(self):
        """Computes the amount of memory required for each of the components. Note that this is not
        necessarily the same as the number of parameters due to some architecture design
        constraints.

        .. math::

            K_{MT} = c \\cdot 2^{\\lceil \\log_2\\left(k_xk_y\\right) \\rceil + \\lceil \\log_2\\left(f\\right) \\rceil}

        .. math::

            N_{MT} = f \\cdot 2^{ \\lceil \\log_2\\left(f_y\\right) \\rceil + \\lceil \\log_2\\left(f_x\\right) \\rceil }

        Returns
        -------
        A dictionary with keys kernel, neuron and bias and the corresponding memory sizes
        """
        summary = self.summary()
        f, c, h, w = summary["kernel"]
        f, neuron_height, neuron_width = self._get_conv_output_shape()        # neuron layer output has the same shape as the convolution layer ouput.

        return {
            "kernel": c * pow(2, np.ceil(np.log2(h * w)) + np.ceil(np.log2(f))),
            "neuron": f
            * pow(2, np.ceil(np.log2(neuron_height)) + np.ceil(np.log2(neuron_width))),
            "bias": 0 if self.conv.bias is None else len(self.conv.bias),
        }
    
    ####################################################### Private Methods #######################################################

    def _convert_linear_to_conv(self, lin: nn.Linear, layer_data: dict) -> Tuple[nn.Conv2d, Tuple[int, int, int]]:
        """ Convert Linear layer to Conv2d.

        Parameters
        ----------
        - lin (nn.Linear): linear layer to be converted.

        Returns
        -------
        - nn.Conv2d: convolutional layer equivalent to `lin`.
        - input_shape (tuple): the tensor shape the layer expects.
        """
        # this flags the necessity to update the I/O shape pre-computed for each of the original layers being compressed within a `DynapcnnLayer` instance.
        self._lin_to_conv_conversion = True

        input_shape = layer_data['input_shape']

        in_chan, in_h, in_w = input_shape

        if lin.in_features != in_chan * in_h * in_w:
            raise ValueError("Shapes don't match.")

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

        return layer, input_shape
    
    def _get_conv_output_shape(self) -> Tuple[int, int, int]:
        """ Computes the output dimensions of `conv_layer`.

        Returns
        ----------
        - output dimensions (tuple): a tuple describing `(output channels, height, width)`.
        """
        # get the layer's parameters.

        out_channels = self.conv.out_channels
        kernel_size = self.conv.kernel_size
        stride = self.conv.stride
        padding = self.conv.padding
        dilation = self.conv.dilation

        # compute the output height and width.
        out_height = ((self.in_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
        out_width = ((self.in_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1

        return (out_channels, out_height, out_width)
