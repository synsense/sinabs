import torch
from torch import nn
import torch.nn.functional as F
from typing import Dict
import sinabs.layers as sl
from warnings import warn
from .discretize import discretize_conv_spike_
from copy import deepcopy


class SumPool2d(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        if isinstance(size, int):
            self.factor = size ** 2
        else:
            self.factor = size[0] * size[1]

    def forward(self, input):
        return self.factor * F.avg_pool2d(
            input, kernel_size=self.size, stride=self.size
        )


class SpeckLayer(nn.Module):
    def __init__(self, conv, pool, spk, in_shape, discretize=True, rescale_weights=1):
        super().__init__()

        self.config_dict = {}
        self.input_shape = in_shape

        if isinstance(conv, nn.Linear):
            conv = self.convert_linear_to_conv(conv)
        else:
            conv = deepcopy(conv)
        spk = deepcopy(spk)
        if discretize:
            # int conversion is done while writing the config.
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        self.conv(conv, rescale_weights=rescale_weights)
        self.pool(pool)
        self.spk(spk)

        self.config_dict["dimensions"].update(self.get_dimensions(in_shape))

        self.output_shape = (
            self.config_dict["dimensions"]["output_feature_count"],
            self.config_dict["dimensions"]["output_size_x"],
            self.config_dict["dimensions"]["output_size_y"],
        )

    def get_dimensions(self, in_shape):
        dimensions = {}

        dims = self.config_dict["dimensions"]

        dimensions["channel_count"] = in_shape[0]
        dimensions["input_size_y"] = in_shape[1]
        dimensions["input_size_x"] = in_shape[2]
        # dimensions["output_feature_count"] already done in conv2d_to_dict
        dimensions["output_size_x"] = (
            (dimensions["input_size_x"] - dims["kernel_size"] + 2 * dims["padding_x"])
            // dims["stride_x"]
            + 1
        ) // self.config_dict["Pooling"]
        dimensions["output_size_y"] = (
            (dimensions["input_size_y"] - dims["kernel_size"] + 2 * dims["padding_y"])
            // dims["stride_y"]
            + 1
        ) // self.config_dict["Pooling"]

        return dimensions

    def convert_linear_to_conv(self, lin):
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

    @staticmethod
    def spklayer_to_dict(layer: sl.SpikingLayer) -> Dict:
        # - Neuron states
        if layer.state.dim() == 1:
            # this should happen when the state is tensor([0.]), which is the
            # Sinabs default for non-initialized networks. We check that and
            # then we assign no initial neuron state to Speck.
            assert len(layer.state) == 1
            assert layer.state.item() == 0.0
            neurons_state = None
        elif layer.state.dim() == 2:
            # this happens when we had a linear layer turned to conv
            layer.state = layer.state.unsqueeze(-1).unsqueeze(-1)
            layer.activations = layer.activations.unsqueeze(-1).unsqueeze(-1)
            neurons_state = layer.state.int().tolist()
        elif layer.state.dim() == 4:
            # 4-dimensional states should be the norm.
            neurons_state = layer.state.transpose(2, 3).int().tolist()
        else:
            raise ValueError("Current state of spiking layer not understood.")

        # - Resetting vs returning to 0
        return_to_zero = layer.membrane_subtract is not None
        # if return_to_zero and layer.membrane_reset != 0:
        #     warn(
        #         f"SpikingConv2dLayer `{layer.layer_name}`: Resetting of membrane potential is always to 0."
        #     )
        if (not return_to_zero) and layer.membrane_subtract != layer.threshold:
            warn(
                f"SpikingConv2dLayer `{layer.layer_name}`: Subtraction of membrane potential is always by high threshold."
            )

        layer_params = dict(
            return_to_zero=return_to_zero,
            threshold_high=int(layer.threshold),
            threshold_low=int(layer.threshold_low),
            monitor_enable=True,  # Yes or no?
            # leak_enable=layer.bias,  # TODO
        )

        return {
            "layer_params": layer_params,
            "neurons_state": neurons_state,
        }

    @staticmethod
    def conv2d_to_dict(layer: nn.Conv2d) -> Dict:
        """
        spiking_conv2d_to_dict - Extract a dictionary with parameters from a `Conv2d`
                                 so that they can be written to a Speck configuration.
        :param layer:   Conv2d whose parameters should be extracted
        :return:    Dict    Parameters of `layer`
        """
        # - Layer dimension parameters
        dimensions = {}

        # - Padding
        dimensions["padding_x"], dimensions["padding_y"] = layer.padding

        # - Stride
        dimensions["stride_x"], dimensions["stride_y"] = layer.stride

        # - Kernel size
        dimensions["kernel_size"] = layer.kernel_size[0]
        if dimensions["kernel_size"] != layer.kernel_size[1]:
            raise ValueError("Conv2d: Kernel must have same height and width.")

        # - Input and output shapes
        dimensions["output_feature_count"] = layer.out_channels

        # - Weights and biases
        if layer.bias is not None:
            weights, biases = layer.parameters()
        else:
            (weights,) = layer.parameters()
            biases = torch.zeros(layer.out_channels)

        # Transpose last two dimensions of weights to match cortexcontrol
        weights = weights.transpose(2, 3)

        return {
            "dimensions": dimensions,
            "weights": weights.int().tolist(),
            "biases": biases.int().tolist(),
        }

    def conv(self, conv, rescale_weights=1):
        self._conv_layer = conv
        if rescale_weights != 1:
            conv.weight.data = (conv.weight / rescale_weights).clone().detach()
        self.config_dict.update(self.conv2d_to_dict(conv))

    def pool(self, pool):
        if pool is not None and pool > 1:
            self._pool_layer = SumPool2d(size=pool)
            self.config_dict["Pooling"] = pool
        else:
            self._pool_layer = None
            self.config_dict["Pooling"] = 1  # TODO is this ok for no pooling?

    def spk(self, spk):
        self._spk_layer = spk
        self.config_dict.update(self.spklayer_to_dict(spk))

    def forward(self, x):
        # print("Input to Speck Layer", x.shape)
        x = self._conv_layer(x)
        # print("After convolution", x.shape)
        x = self._spk_layer(x)
        # print("After spiking", x.shape)
        if self._pool_layer:
            x = self._pool_layer(x)
            # print("After pooling", x.shape)
        return x
