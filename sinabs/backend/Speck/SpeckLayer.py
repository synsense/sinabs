import torch
from torch import nn
from typing import Dict
import sinabs.layers as sl
from warnings import warn
from .discretize import discretize_conv_spike


class SpeckLayer(nn.Module):
    def __init__(self, conv, pool, spk, in_shape, discretize=True):
        super().__init__()

        self.config_dict = {}
        self.input_shape = in_shape

        if isinstance(conv, nn.Linear):
            conv = self.convert_linear_to_conv(conv)
        if discretize:
            conv, spk = discretize_conv_spike(conv, spk, to_int=False)
        self.conv(conv)
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
        dimensions["output_size_x"] = ((
            dimensions["input_size_x"] - dims["kernel_size"] + 2 * dims["padding_x"]
        ) // dims["stride_x"] + 1) // self.config_dict["Pooling"]
        dimensions["output_size_y"] = ((
            dimensions["input_size_y"] - dims["kernel_size"] + 2 * dims["padding_y"]
        ) // dims["stride_y"] + 1) // self.config_dict["Pooling"]

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
        if layer.state is not None:
            if len(layer.state.shape) == 2:
                # this happens when we had a linear layer turned to conv
                layer.state = layer.state.unsqueeze(-1).unsqueeze(-1)
                layer.activations = layer.activations.unsqueeze(-1).unsqueeze(-1)
            neurons_state = layer.state.transpose(2, 3).int().tolist()
        else:
            neurons_state = None

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
            weights, = layer.parameters()
            biases = torch.zeros(layer.out_channels)

        # Transpose last two dimensions of weights to match cortexcontrol
        weights = weights.transpose(2, 3)

        return {
            "dimensions": dimensions,
            "weights": weights.int().tolist(),
            "biases": biases.int().tolist(),
        }

    def conv(self, conv):
        self._conv_layer = conv
        self.config_dict.update(self.conv2d_to_dict(conv))

    def pool(self, pool):
        if pool is not None:
            self._pool_layer = nn.AvgPool2d(kernel_size=pool, stride=pool)
            self.config_dict["Pooling"] = pool
        else:
            self._pool_layer = lambda x: x  # do nothing
            self.config_dict["Pooling"] = 1  # check, is this ok for no pooling?

    def spk(self, spk):
        if spk is not None:
            self._spk_layer = spk
            self.config_dict.update(self.spklayer_to_dict(spk))
        else:
            self._spk_layer = nn.ReLU()  # TODO temporary

    def forward(self, x):
        print("Input to Speck Layer", x.shape)
        x = self._conv_layer(x)
        print("After convolution", x.shape)
        x = self._spk_layer(x)
        print("After spiking", x.shape)
        x = self._pool_layer(x)
        print("After pooling", x.shape)
        return x
