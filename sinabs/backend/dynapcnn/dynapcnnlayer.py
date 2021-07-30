import torch
from torch import nn
import numpy as np
from typing import Dict, Tuple, Optional, Union
import sinabs.layers as sl
from warnings import warn
from .discretize import discretize_conv_spike_
from copy import deepcopy

from .dvslayer import expand_to_pair


class DynapcnnLayer(nn.Module):
    """
    Create a DynapcnnLayer object representing a dynapcnn layer.

    Requires a convolutional layer, a sinabs spiking layer and an optional
    pooling value. The layers are used in the order conv -> spike -> pool.

    Parameters
    ----------
        conv: torch.nn.Conv2d or torch.nn.Linear
            Convolutional or linear layer (linear will be converted to convolutional)
        spk: sinabs.layers.SpikingLayer
            Sinabs spiking layer
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
            spk: sl.SpikingLayer,
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
            if spk.state.dim() == 1 and len(spk.state) == 1 and spk.state.item() == 0.0:
                # Layer is uninitialized. Leave it as it is
                pass
            else:
                # Expand dims
                spk.state = spk.state.unsqueeze(-1).unsqueeze(-1)
                spk.activations = spk.activations.unsqueeze(-1).unsqueeze(-1)
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

    def get_config_dict(self):
        config_dict = {}
        config_dict["destinations"] = [{}, {}]

        # Update the dimensions
        channel_count, input_size_y, input_size_x = self.input_shape
        dimensions = {"input_shape": {}, "output_shape": {}}
        dimensions["input_shape"]["size"] = {"x": input_size_x, "y": input_size_y}
        dimensions["input_shape"]["feature_count"] = channel_count

        # dimensions["output_feature_count"] already done in conv2d_to_dict
        (f, h, w) = self.get_neuron_shape()
        dimensions["output_shape"]["size"] = {}
        dimensions["output_shape"]["feature_count"] = f
        dimensions["output_shape"]["size"]["x"] = w
        dimensions["output_shape"]["size"]["y"] = h
        dimensions["padding"] = {"x": self.conv_layer.padding[1], "y": self.conv_layer.padding[0]}
        dimensions["stride"] = {"x": self.conv_layer.stride[1], "y": self.conv_layer.stride[0]}
        dimensions["kernel_size"] = self.conv_layer.kernel_size[0]

        if dimensions["kernel_size"] != self.conv_layer.kernel_size[1]:
            raise ValueError("Conv2d: Kernel must have same height and width.")
        config_dict["dimensions"] = dimensions
        # Update parameters from convolution
        if self.conv_layer.bias is not None:
            (weights, biases) = self.conv_layer.parameters()
        else:
            (weights,) = self.conv_layer.parameters()
            biases = torch.zeros(self.conv_layer.out_channels)
        weights = weights.transpose(2, 3)  # Need this to match samna convention
        config_dict["weights"] = weights.int().tolist()
        config_dict["weights_kill_bit"] = torch.zeros_like(weights).bool().tolist()
        config_dict["biases"] = biases.int().tolist()
        config_dict["biases_kill_bit"] = torch.zeros_like(biases).bool().tolist()
        config_dict["leak_enable"] = biases.bool().any()

        # Update parameters from the spiking layer

        # - Neuron states
        if (
                self.spk_layer.state.dim() == 1
                and len(self.spk_layer.state) == 1
                and self.spk_layer.state.item() == 0.0
        ):
            # this should happen when the state is tensor([0.]), which is the
            # Sinabs default for non-initialized networks. We check that and
            # then we assign no initial neuron state to DYNAP-CNN.
            f, h, w = self.get_neuron_shape()
            neurons_state = torch.zeros(f, w, h)
        elif self.spk_layer.state.dim() == 3:
            # 3-d is the norm when there is no batch dimension in sinabs
            neurons_state = self.spk_layer.state.transpose(1, 2)
        elif self.spk_layer.state.dim() == 4:
            # 4-dimensional states should be the norm when there is a batch dim
            neurons_state = self.spk_layer.state.transpose(2, 3)[0]
        else:
            raise ValueError(
                f"Current state (shape: {self.spk_layer.state.shape}) of spiking layer not understood."
            )

        # - Resetting vs returning to 0
        return_to_zero = self.spk_layer.membrane_reset
        if (not return_to_zero) and self.spk_layer.membrane_subtract != self.spk_layer.threshold:
            warn(
                "SpikingConv2dLayer: Subtraction of membrane potential is always by high threshold."
            )
        config_dict.update({
            "return_to_zero": return_to_zero,
            "threshold_high": int(self.spk_layer.threshold),
            "threshold_low": int(self.spk_layer.threshold_low),
            "monitor_enable": False,
            "neurons_initial_value": neurons_state.int().tolist(),
            "neurons_value_kill_bit": torch.zeros_like(neurons_state).bool().tolist(),
        })
        # Update parameters from pooling
        if self.pool_layer is not None:
            config_dict["destinations"][0]["pooling"] = expand_to_pair(self.pool_layer.kernel_size)[0]
            config_dict["destinations"][0]["enable"] = True
        else:
            pass
        return config_dict

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
