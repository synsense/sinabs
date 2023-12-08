import sinabs.activation
import torch
from torch import nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
import sinabs.layers as sl
from warnings import warn
from .discretize import discretize_conv_spike_
from copy import deepcopy
from .dvs_layer import expand_to_pair


class DynapcnnLayer(nn.Module):
    """
    Create a DynapcnnLayer object representing a dynapcnn layer.

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
            conv: nn.Conv2d,
            spk: sl.IAFSqueeze,
            in_shape: Tuple[int, int, int],
            pool: Optional[sl.SumPool2d] = None,
            discretize: bool = True,
            rescale_weights: int = 1,
    ):
        super().__init__()

        self.input_shape = in_shape

        spk = deepcopy(spk)
        if isinstance(conv, nn.Linear):
            conv = self._convert_linear_to_conv(conv)
            if spk.is_state_initialised():
                # Expand dims
                spk.v_mem = spk.v_mem.data.unsqueeze(-1).unsqueeze(-1)
        else:
            conv = deepcopy(conv)

        if rescale_weights != 1:
            # this has to be done after copying but before discretizing
            conv.weight.data = (conv.weight / rescale_weights).clone().detach()

        self.discretize = discretize
        if discretize:
            # int conversion is done while writing the config.
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        self.conv_layer = conv
        self.spk_layer = spk
        if pool is not None:
            if pool.kernel_size[0] != pool.kernel_size[1]:
                raise ValueError("Only square kernels are supported")
            self.pool_layer = deepcopy(pool)
        else:
            self.pool_layer = None

    def _convert_linear_to_conv(self, lin: nn.Linear) -> nn.Conv2d:
        """
        Convert Linear layer to Conv2d.

        Parameters
        ----------
            lin: nn.Linear
                Linear layer to be converted

        Returns
        -------
            nn.Conv2d
                Convolutional layer equivalent to `lin`.
        """

        in_chan, in_h, in_w = self.input_shape

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

    def get_neuron_shape(self) -> Tuple[int, int, int]:
        """
        Return the output shape of the neuron layer

        Returns
        -------
        features, height, width
        """

        def get_shape_after_conv(layer: nn.Conv2d, input_shape):
            (ch_in, h_in, w_in) = input_shape
            (kh, kw) = expand_to_pair(layer.kernel_size)
            (pad_h, pad_w) = expand_to_pair(layer.padding)
            (stride_h, stride_w) = expand_to_pair(layer.stride)

            def out_len(in_len, k, s, p):
                return (in_len - k + 2 * p) // s + 1

            out_h = out_len(h_in, kh, stride_h, pad_h)
            out_w = out_len(w_in, kw, stride_w, pad_w)
            ch_out = layer.out_channels
            return ch_out, out_h, out_w

        conv_out_shape = get_shape_after_conv(self.conv_layer, input_shape=self.input_shape)
        return conv_out_shape

    def get_output_shape(self) -> Tuple[int, int, int]:
        neuron_shape = self.get_neuron_shape()
        # this is the actual output shape, including pooling
        if self.pool_layer is not None:
            pool = expand_to_pair(self.pool_layer.kernel_size)
            return (
                neuron_shape[0],
                neuron_shape[1] // pool[0],
                neuron_shape[2] // pool[1],
            )
        else:
            return neuron_shape

    def summary(self) -> dict:
        return {
            "pool": None if self.pool_layer is None else list(self.pool_layer.kernel_size),
            "kernel": list(self.conv_layer.weight.data.shape),
            "neuron": self.get_neuron_shape(),
        }

    def memory_summary(self):
        """
        Computes the amount of memory required for each of the components.
        Note that this is not necessarily the same as the number of parameters due to some architecture design constraints.

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
            "neuron": f * pow(2, np.ceil(np.log2(neuron_height)) + np.ceil(np.log2(neuron_width))),
            "bias": 0 if self.conv_layer.bias is None else len(self.conv_layer.bias),
        }

    def forward(self, x):
        """Torch forward pass."""
        x = self.conv_layer(x)
        x = self.spk_layer(x)
        if self.pool_layer is not None:
            x = self.pool_layer(x)
        return x

    def zero_grad(self, set_to_none: bool = False) -> None:
        return self.spk_layer.zero_grad(set_to_none)