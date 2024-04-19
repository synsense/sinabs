from copy import deepcopy
from typing import Dict, Optional, Tuple, Union
from warnings import warn

import numpy as np
import torch
from torch import nn

import sinabs.activation
import sinabs.layers as sl

from .discretize import discretize_conv_spike_
from .dvs_layer import expand_to_pair


class DynapcnnLayer(nn.Module):
    """Create a DynapcnnLayer object representing a dynapcnn layer.

    Requires a convolutional layer, a sinabs spiking layer and an optional
    pooling value. The layers are used in the order conv -> spike -> pool.

    Parameters
    ----------
        conv: torch.nn.Conv2d or torch.nn.Linear
            Convolutional or linear layer (linear will be converted to convolutional)
        spk: sinabs.layers.IAFSqueeze
            Sinabs IAF layer
        in_shape: tuple of int
            The input shape, needed to create dynapcnn configs if the network does not
            contain an input layer. Convention: (features, height, width)
        pool: int or None
            Integer representing the sum pooling kernel and stride. If `None`, no
            pooling will be applied.
        discretize: bool
            Whether to discretize parameters.
        rescale_weights: int
            Layer weights will be divided by this value.
    """

    def __init__(
        self,
        dcnnl_data: dict, 
        discretize: bool,
        rescale_weights: int = 1,       # TODO remove.
    ):
        super().__init__()
        """
        ...

        TODO
            1) need to figure out how to apply 'rescale_weights' since there are more than two poolings.
            2) currently there's no way the forward would work since there are more than two poolings.
        """

        #### NEW WAY #######################################################################

        conv = None
        conv_node_id = None

        spk = None
        spk_node_id = None

        pool = []
        pool_node_id = []

        for key, value in dcnnl_data.items():
            if isinstance(key, int):
                # value has data pertaining a node (torch/sinabs layer).
                if isinstance(value['layer'], sl.IAFSqueeze):
                    spk = value['layer']
                    spk_node_id = key
                elif isinstance(value['layer'], nn.Linear) or isinstance(value['layer'], nn.Conv2d):
                    conv = value['layer']
                    conv_node_id = key
                elif isinstance(value['layer'], sl.SumPool2d):
                    pool.append(value['layer'])
                    pool_node_id.append(key)
                else:
                    raise ValueError(f'Node {key} has not valid layer associated with it.')
                
        if not conv:
            raise ValueError(f'Convolution layer not present.')
        
        if not spk:
            raise ValueError(f'Spiking layer not present.')
        
        spk = deepcopy(spk)
        if spk.is_state_initialised():
                spk.v_mem = spk.v_mem.data.unsqueeze(-1).unsqueeze(-1)      # expand dims.

        if isinstance(conv, nn.Linear):
            conv = self._convert_linear_to_conv(conv, dcnnl_data[conv_node_id])
        else:
            conv = deepcopy(conv)

        # TODO have to consider that two poolings might be projecting to this conv (these lines of code are deprecated).
        # this weight rescale comes from the node projecting into this 'conv' node.
        if rescale_weights != 1:
            # this has to be done after copying but before discretizing
            conv.weight.data = (conv.weight / rescale_weights).clone().detach()

        # int conversion is done while writing the config.
        if discretize:
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        # consolidate layers.
        self.conv_layer = conv
        self.spk_layer = spk
        self.pool_layer = []
        if len(pool) != 0:
            for plyr in pool:
                if plyr.kernel_size[0] != plyr.kernel_size[1]:
                    raise ValueError("Only square kernels are supported")
                self.pool_layer.append(deepcopy(plyr))

    def _convert_linear_to_conv(self, lin: nn.Linear, layer_data: dict) -> nn.Conv2d:
        """Convert Linear layer to Conv2d.

        Parameters
        ----------
            lin: nn.Linear
                Linear layer to be converted

        Returns
        -------
            nn.Conv2d
                Convolutional layer equivalent to `lin`.
        """

        #   TODO linear layers are convered to conv so the input shapes in the original
        # mapper has to be updated accordingly. Another problem is that the input shape
        # after a flatten layer won't match the fact that there's a convolution output before
        # the flatten.
        input_shape = tuple(list(layer_data['input_shape'])[1:-1])  # removing the batch dimension.

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

        return layer
    
    def zero_grad(self, set_to_none: bool = False) -> None:
        return self.spk_layer.zero_grad(set_to_none)
    
    ########################################################################################

    def summary(self) -> dict:                                      # TODO deprecated.
        return {
            "pool": (
                None if self.pool_layer is None else list(self.pool_layer.kernel_size)
            ),
            "kernel": list(self.conv_layer.weight.data.shape),
            "neuron": self.get_neuron_shape(),
        }

    def memory_summary(self):                                       # TODO deprecated.
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
        f, neuron_height, neuron_width = self.get_neuron_shape()

        return {
            "kernel": c * pow(2, np.ceil(np.log2(h * w)) + np.ceil(np.log2(f))),
            "neuron": f
            * pow(2, np.ceil(np.log2(neuron_height)) + np.ceil(np.log2(neuron_width))),
            "bias": 0 if self.conv_layer.bias is None else len(self.conv_layer.bias),
        }

    def forward(self, x):                                           # TODO deprecated.
        """Torch forward pass."""
        x = self.conv_layer(x)
        x = self.spk_layer(x)
        if self.pool_layer is not None:
            x = self.pool_layer(x)
        return x
