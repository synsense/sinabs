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

        self.in_shape           = in_shape
        self.pool               = pool
        self.discretize         = discretize
        self.rescale_weights    = rescale_weights

        spk = deepcopy(spk)

        if isinstance(conv, nn.Linear):
            conv = self._convert_linear_to_conv(conv)
            if spk.is_state_initialised():
                # Expand dims
                spk.v_mem = spk.v_mem.data.unsqueeze(-1).unsqueeze(-1)
        else:
            conv = deepcopy(conv)

        if self.rescale_weights != 1:
            # this has to be done after copying but before discretizing
            conv.weight.data = (conv.weight / self.rescale_weights).clone().detach()

        # int conversion is done while writing the config.
        if self.discretize:
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        if discretize:
            # int conversion is done while writing the config.
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        self.conv               = conv
        self.spk                = spk
        
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
        - conv_out_shape (tuple): formatted as (features, height, width).
        """
        # same as the convolution's output.
        return self.conv_out_shape

    def summary(self) -> dict:
        """ Returns a summary of the convolution's/pooling's kernel sizes and the output shape of the spiking layer."""
        # TODO I can't see pooling being used in checking memory constraints by the builder so I'm ignoring for now the fact that multiple pooling could exist.
        
        _pool = None

        # @TODO POSSIBLE INCONSISTENCY: if the `SumPool2d` is the result of a conversion from `AvgPool2d` then `SumPool2d.kernel_size` 
        # is of type tuple, otherwise it is an int. 
        if self._pool_lyrs:
            # @TODO ignoring for now that there could be multiple poolings (just use the first one).
            if isinstance(self._pool_lyrs[next(iter(self._pool_lyrs))].kernel_size, tuple):
                _pool = list(self._pool_lyrs[next(iter(self._pool_lyrs))].kernel_size)
            elif isinstance(self._pool_lyrs[next(iter(self._pool_lyrs))].kernel_size, int):
                _pool = [self._pool_lyrs[next(iter(self._pool_lyrs))].kernel_size, self._pool_lyrs[next(iter(self._pool_lyrs))].kernel_size]
            else:
                raise ValueError('Type of `self.pool_layer[0].kernel_size` not understood.')

        return {
            "pool": (_pool),
            "kernel": list(self.conv_layer.weight.data.shape),
            "neuron": self.conv_out_shape,                          # neuron layer output has the same shape as the convolution layer ouput.
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
        f, neuron_height, neuron_width = self.conv_out_shape        # neuron layer output has the same shape as the convolution layer ouput.

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

        spk = deepcopy()
        out_channels = self.conv.out_channels

        spk = deepcopy()
        kernel_size = self.conv.kernel_size
        stride = self.conv.stride
        padding = self.conv.padding
        dilation = self.conv.dilation

        # compute the output height and width.
        out_height = ((self.in_shape[1] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
        out_width = ((self.in_shape[2] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1

        return (out_channels, out_height, out_width)