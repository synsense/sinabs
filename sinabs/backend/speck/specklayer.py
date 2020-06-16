import torch
from torch import nn
from typing import Dict, Tuple, Optional
import sinabs.layers as sl
from warnings import warn
from .discretize import discretize_conv_spike_
from copy import deepcopy


class SpeckLayer(nn.Module):
    """Torch module that reproduces the behaviour of a speck layer."""

    def __init__(
        self,
        conv: nn.Conv2d,
        spk: sl.SpikingLayerBPTT,
        in_shape: Tuple[int],
        pool: Optional[bool] = None,
        discretize: bool = True,
        rescale_weights: int = 1,
    ):
        """
        Create a SpeckLayer object representing a speck layer.

        Requires a convolutional layer, a sinabs spiking layer and an optional
        pooling value. The layers are used in the order conv -> spike -> pool.

        Parameters
        ----------
            conv: torch.nn.Conv2d or torch.nn.Linear
                Convolutional or linear layer (linear will be converted to convolutional)
            spk: sinabs.layers.SpikingLayer
                Sinabs spiking layer
            in_shape: tuple of int
                The input shape, needed to create speck configs if the network does not
                contain an input layer. Convention: (features, height, width)
            pool: int or None
                Integer representing the sum pooling kernel and stride. If `None`, no
                pooling will be applied.
            discretize: bool
                Whether to discretize parameters.
            rescale_weights: int
                Layer weights will be divided by this value.
        """
        super().__init__()

        self.config_dict = {}
        self.input_shape = in_shape

        if isinstance(conv, nn.Linear):
            conv = self._convert_linear_to_conv(conv)
        else:
            conv = deepcopy(conv)
        spk = deepcopy(spk)

        if rescale_weights != 1:
            # this has to be done after copying but before rescaling
            conv.weight.data = (conv.weight / rescale_weights).clone().detach()

        if discretize:
            # int conversion is done while writing the config.
            conv, spk = discretize_conv_spike_(conv, spk, to_int=False)

        self._conv(conv)
        self._pool(pool)
        self._spk(spk)

        self.config_dict["dimensions"].update(self._get_dimensions(in_shape))

        self.output_shape = (
            self.config_dict["dimensions"]["output_feature_count"],
            self.config_dict["dimensions"]["output_size_x"],
            self.config_dict["dimensions"]["output_size_y"],
        )

    def _get_dimensions(self, in_shape: Tuple[int]) -> Dict[str, int]:
        """
        Calculate layer output dimensions from input shape.

        Parameters
        ----------
            in_shape: tuple of int
                Input shape. Convention: (features, height, width)

        Returns
        -------
            dict
                Input and output dimensions
        """

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

    @staticmethod
    def _spklayer_to_dict(layer: sl.SpikingLayerBPTT) -> Dict:
        """
        Extract parameters of spiking layer into dict.

        Parameters
        ----------
            layer: sl.SpikingLayerBPTT

        Returns
        -------
            dict

        """
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
    def _conv2d_to_dict(layer: nn.Conv2d) -> Dict:
        """
        Extract a dictionary with parameters from a `Conv2d` so that they can be
        written to a Speck configuration.

        Parameters
        ----------
            layer: nn.Conv2d
                Convolutional layer whose parameters should be extracted

        Returns
        -------
            dict
                Parameters of `layer`
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

    def _conv(self, conv):
        self._conv_layer = conv
        self.config_dict.update(self._conv2d_to_dict(conv))

    def _pool(self, pool):
        if pool is not None and pool > 1:
            self._pool_layer = sl.SumPool2d(kernel_size=pool, stride=pool)
            self.config_dict["Pooling"] = pool
        else:
            self._pool_layer = None
            self.config_dict["Pooling"] = 1  # TODO is this ok for no pooling?

    def _spk(self, spk):
        self._spk_layer = spk
        self.config_dict.update(self._spklayer_to_dict(spk))

    def forward(self, x):
        """Torch forward pass."""
        # print("Input to Speck Layer", x.shape)
        x = self._conv_layer(x)
        # print("After convolution", x.shape)
        x = self._spk_layer(x)
        # print("After spiking", x.shape)
        if self._pool_layer:
            x = self._pool_layer(x)
            # print("After pooling", x.shape)
        return x
