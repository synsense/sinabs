# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

from copy import deepcopy
from typing import Dict, Callable, Tuple, Union, List

import numpy as np
import torch
from torch import nn

import sinabs.activation
import sinabs.layers as sl

from .discretize import discretize_conv_spike_

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

        # int conversion is done while writing the config.
        if discretize:
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        self.conv               = conv
        self.spk                = spk
        self.in_shape           = in_shape
        self.pool               = pool
        self.discretize         = discretize
        self.rescale_weights    = rescale_weights
        self.conv_out_shape     = self._get_conv_output_shape()

        self._pool_lyrs         = self._make_pool_layers()                                 # creates SumPool2d layers from `pool`.
        

    ####################################################### Public Methods #######################################################
    
    def forward(self, x):
        """Torch forward pass.

        ...
        """

        returns = []
        
        x = self.conv(x)
        x = self.spk(x)

        for pool in self.pool:
            if pool == 1:
                # no pooling is applied.
                returns.append(x)
            else:
                # sum pooling of `(pool, pool)` is applied.
                pool_out = self._pool_lyrs[pool](x)
                returns.append(pool_out)

        return tuple(returns)
    
    def get_neuron_shape(self) -> Tuple[int, int, int]:
        """Return the output shape of the neuron layer.

        Returns
        -------
        features, height, width
        """
        # same as the convolution's output.
        return self.conv_out_shape
    
    ####################################################### Private Methods #######################################################

    def _make_pool_layers(self) -> Dict[int, sl.SumPool2d]:
        """ Creates a `sl.SumPool2d` for each entry in `self.pool` greater than one.

        Note: the "kernel size" (values > 1) in self.pool is by default used to set the stride of the pooling layer.

        Returns
        -------
        - pool_lyrs (dict): the `key` is a value grather than 1 in `self.pool`, with the `value` being the `sl.SumPool2d` it represents.
        """

        pool_lyrs = {}

        # validating if pool are integers
        for item in self.pool:
            if not isinstance(item, int):
                raise ValueError(f"Item '{item}' in `pool` is not an integer.")

        # create layers form pool list.
        for kernel_s in self.pool:

            if kernel_s != 1:

                pooling = (kernel_s, kernel_s)
                cumulative_pooling = (1, 1)

                # compute cumulative pooling.
                cumulative_pooling = (
                    cumulative_pooling[0] * pooling[0],
                    cumulative_pooling[1] * pooling[1],
                )

                # create SumPool2d layer.
                pool_lyrs[kernel_s] = sl.SumPool2d(cumulative_pooling)

        return pool_lyrs
    
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