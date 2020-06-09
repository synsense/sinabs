from warnings import warn
from copy import deepcopy

try:
    from samna.speck.configuration import SpeckConfiguration, CNNLayerConfig
except (ImportError, ModuleNotFoundError):
    SAMNA_AVAILABLE = False
else:
    SAMNA_AVAILABLE = True
from .SpeckLayer import SpeckLayer, SumPool2d

import torch.nn as nn
import torch
import sinabs.layers as sl
import sinabs
from typing import Dict, Tuple, Union, Optional


class SpeckCompatibleNetwork(nn.Module):
    def __init__(
        self,
        snn: Union[nn.Module, sinabs.Network],
        input_shape: Optional[Tuple[int]] = None,
        dvs_input: bool = True,
        discretize: bool = True,
    ) -> Dict:
        """
        Given a sinabs spiking network, prepare a speck-compatible network.
        In particular, the following will be done:
        - multiple pooling layers in a row will be consolidated into one
        - checks are performed on layer hyperparameter compatibility with speck
        - checks are performed on network structure compatibility with speck
        - linear layers are turned into convolutional layers
        - dropout layers are ignored
        - weights, biases and thresholds are discretized according to speck requirements

        :param snn: sinabs.Network or torch.nn.Module instance
        """
        super().__init__()

        # this holds the SpeckLayer objects which can be used for testing
        # and also deal with single-layer-level configuration issues
        self.compatible_layers = []

        # TODO: Currently only spiking seq. models are supported
        try:
            layers = list(snn.spiking_model.seq)
        except AttributeError:
            raise ValueError("`snn` must contain a sequential spiking model.")

        # index that goes over the layers of the input network
        i_layer = 0
        # used to carry pooling info to next conv, to rescale weights due to
        # the difference between sum and average pooling
        rescaling_from_pooling = 1

        # - Input to start with
        if isinstance(layers[0], sl.InputLayer):
            input_layer = deepcopy(layers[0])
            input_shape = input_layer.output_shape
            self.compatible_layers.append(input_layer)
            i_layer += 1
        elif input_shape is None:
            raise ValueError(
                "`input_shape` must be provided if first layer is not `InputLayer`."
            )
        self._dvs_input = dvs_input
        self._external_input_shape = input_shape
        self._discretize = discretize

        # - Iterate over layers from model
        while i_layer < len(layers):
            # Layer to be ported to Speck
            lyr_curr = layers[i_layer]

            if isinstance(lyr_curr, (nn.Conv2d, nn.Linear)):
                # Linear and Conv layers are dealt with in the same way.
                i_next, input_shape, rescaling_from_pooling = self._handle_conv2d_layer(
                    layers[i_layer:], input_shape, rescaling_from_pooling
                )

                if i_next is None:
                    # TODO: How to route to readout layer? Does destination need to be set?
                    break
                else:
                    # Add 2 to i_layer to go to next layer, + i_next for number
                    # of consolidated pooling layers
                    i_layer += i_next + 2

            elif isinstance(lyr_curr, nn.AvgPool2d):
                pooling, i_next = self.consolidate_pooling(layers[i_layer:], dvs=True)
                self.compatible_layers.append(SumPool2d(size=pooling))
                rescaling_from_pooling = pooling[0] * pooling[1]

                if i_next is not None:
                    i_layer += i_next
                else:
                    break

            elif isinstance(lyr_curr, (nn.Dropout2d, nn.Flatten)):
                # - Ignore dropout and flatten layers
                i_layer += 1

            elif isinstance(lyr_curr, sl.InputLayer):
                raise TypeError(f"Only first layer can be of type {type(lyr_curr)}.")

            elif isinstance(lyr_curr, sl.iaf_bptt.SpikingLayer):
                raise TypeError(
                    f" Layer of type {type(lyr_curr)} must be preceded by layer of "
                    "type `Conv2d` or `Linear`."
                )

            else:
                raise TypeError(
                    f"Layers of type {type(lyr_curr)} are currently not supported."
                )

        # TODO: Does anything need to be done after iterating over layers?
        # print("Finished configuration of Speck.")

        self.sequence = nn.Sequential(*self.compatible_layers)

    def make_config(self, speck_layers_ordering=range(9)):
        if not SAMNA_AVAILABLE:
            raise ImportError("`samna` does not appear to be installed.")

        config = SpeckConfiguration()

        if self._dvs_input:
            # - Cut DVS output to match output shape of `lyr_curr`
            dvs = config.dvs_layer
            dvs.cut.y = self._external_input_shape[1]
            dvs.cut.x = self._external_input_shape[2]
            # TODO: How to deal with feature count?

        i_layer_speck = 0
        for i, speck_equivalent_layer in enumerate(self.sequence):
            # happens when the network starts with pooling
            if isinstance(speck_equivalent_layer, nn.AvgPool2d):
                # This case can only happen if `self.sequence` starts with a pooling layer
                # or input layer because all other pooling layers should get consolidated.
                # Therefore, assume that input comes from DVS.
                # TODO test
                # Object representing Speck DVS
                dvs = config.dvs_layer
                dvs.pooling.y, dvs.pooling.x = speck_equivalent_layer.kernel_size
                dvs.destinations[0].layer = speck_layers_ordering[i_layer_speck]
                dvs.destinations[0].enable = True

            elif isinstance(speck_equivalent_layer, SpeckLayer):
                # Object representing Speck layer
                speck_layer = config.cnn_layers[speck_layers_ordering[i_layer_speck]]
                # read the configuration dictionary from SpeckLayer
                # and write it to the speck configuration object
                self.write_speck_config(speck_equivalent_layer.config_dict, speck_layer)

                # For now: Sequential model, second destination always disabled
                speck_layer.destinations[1].enable = False

                if i == len(self.sequence) - 1:
                    # last layer
                    speck_layer.destinations[0].enable = False
                else:
                    # Set destination layer
                    speck_layer.destinations[0].layer = speck_layers_ordering[
                        i_layer_speck + 1
                    ]
                    speck_layer.destinations[0].pooling = speck_equivalent_layer.config_dict["Pooling"]
                    speck_layer.destinations[0].enable = True

                i_layer_speck += 1
            else:
                # in our generated network there is a spurious layer...
                # should never happen
                raise TypeError("Unexpected layer in generated network")
        return config

    def _handle_conv2d_layer(self, layers, input_shape, rescaling_from_pooling):
        lyr_curr = layers[0]

        # Next layer needs to be spiking
        try:
            lyr_next = layers[1]
        except IndexError:
            reached_end = True
        else:
            reached_end = False
        if reached_end or not isinstance(lyr_next, sl.iaf_bptt.SpikingLayer):
            raise TypeError("Convolutional layer must be followed by spiking layer.")

        # - Consolidate pooling from subsequent layers
        pooling, i_next = self.consolidate_pooling(layers[2:], dvs=False)

        # The SpeckLayer object knows how to turn the conv-spk-pool trio to
        # a speck layer, and has a forward method, and computes the output shape
        compatible_object = SpeckLayer(
            conv=lyr_curr,
            spk=lyr_next,
            pool=pooling,
            in_shape=input_shape,
            discretize=self._discretize,
            rescale_weights=rescaling_from_pooling,
        )
        # the previous rescaling has been used, the new one is used in the next layer
        rescaling_from_pooling = pooling**2
        # we save this object for future forward passes for testing
        self.compatible_layers.append(compatible_object)
        output_shape = compatible_object.output_shape

        return i_next, output_shape, rescaling_from_pooling

    def forward(self, x):
        self.eval()
        with torch.no_grad():
            return self.sequence(x)

    def write_speck_config(
        self, config_dict: dict, speck_layer: "CNNLayerConfig",
    ):  # -> Tuple[int]:
        # Update configuration of the Speck layer
        # print("Setting dimensions:")
        # pprint(layer_config["dimensions"])
        # print("Setting weights, shape:", np.array(layer_config["weights"]).shape)
        # print("Setting biases, shape:", np.array(layer_config["biases"]).shape)
        speck_layer.dimensions = config_dict["dimensions"]
        speck_layer.weights = config_dict["weights"]
        speck_layer.biases = config_dict["biases"]
        if config_dict["neurons_state"] is not None:
            pass
            # print("Setting state:", layer_config["neurons_state"])
            # TODO unclear why error
            # speck_layer.neurons_initial_value = layer_config["neurons_state"]
        for param, value in config_dict["layer_params"].items():
            # print(f"Setting parameter {param}: {value}")
            setattr(speck_layer, param, value)

    def consolidate_pooling(
        self, layers, dvs: bool
    ) -> Tuple[Union[int, Tuple[int], None], int]:
        """
        consolidate_pooling - Consolidate the first `SumPooling2dLayer`s in `layers`
                              until the first object of different type.
        :param layers:  Iterable, containing `SumPooling2dLayer`s and other objects.
        :param dvs:     bool, if True, x- and y- pooling may be different and a
                              Tuple is returned instead of an integer.
        :return:
            int or tuple, consolidated pooling size. Tuple if `dvs` is true.
            int or None, index of first object in `layers` that is not a
                         `SumPooling2dLayer`, or `None`, if all objects in `layers`
                         are `SumPooling2dLayer`s.
        """

        pooling = [1, 1] if dvs else 1

        for i_next, lyr in enumerate(layers):
            if isinstance(lyr, nn.AvgPool2d):
                # Update pooling size
                new_pooling = self.get_pooling_size(lyr, dvs=dvs)
                if dvs:
                    pooling[0] *= new_pooling[0]
                    pooling[1] *= new_pooling[1]
                else:
                    pooling *= new_pooling
            else:
                # print("Pooling:", pooling)
                # print("Output shape:", input_shape)
                return pooling, i_next

        # If this line is reached, all objects in `layers` are pooling layers.
        # print("Pooling:", pooling)
        # print("Output shape:", input_shape)
        return pooling, None

    def get_pooling_size(
        self, layer: nn.AvgPool2d, dvs: bool
    ) -> Union[int, Tuple[int]]:
        """
        get_pooling_size - Determine the pooling size of a pooling object.
        :param layer:  `AvgPool2d` object
        :param dvs:    bool - If True, pooling does not need to be symmetric.
        :return:
            int or tuple - pooling size. If `dvs` is true, then return a tuple with
                           sizes for y- and x-pooling.
        """
        # Warn if there is non-zero padding.
        # Padding can be either int or tuple of ints
        if isinstance(layer.padding, int):
            warn_padding = layer.padding != 0
        else:
            warn_padding = any(pad != 0 for pad in layer.padding)
        if warn_padding:
            warn(
                f"AvgPool2d `{layer.layer_name}`: Padding is not supported for pooling layers."
            )

        # - Pooling and stride
        pooling = layer.kernel_size
        pooling_y, pooling_x = (
            (pooling, pooling) if isinstance(pooling, int) else pooling
        )

        stride = layer.stride
        stride_y, stride_x = (stride, stride) if isinstance(stride, int) else stride

        if dvs:
            # Check whether pooling and strides match
            if stride_y != pooling_y or stride_x != pooling_x:
                raise ValueError(
                    f"AvgPool2d `{layer.layer_name}`: Stride size must be the same as pooling size."
                )
            return (pooling_y, pooling_x)
        else:
            # Check whether pooling is symmetric
            if pooling_x != pooling_y:
                raise ValueError(
                    f"AvgPool2d `{layer.layer_name}`: Pooling must be symmetric for CNN layers."
                )
            pooling = pooling_x  # Is this the vertical dimension?
            # Check whether pooling and strides match
            if any(stride != pooling for stride in (stride_x, stride_y)):
                raise ValueError(
                    f"AvgPool2d `{layer.layer_name}`: Stride size must be the same as pooling size."
                )
            return pooling


# def identity_dimensions(input_shape: Tuple[int]) -> sd.configuration.CNNLayerDimensions:
#     """
#     identity_dimensions - Return `CNNLayerDimensions` for Speck such that the layer
#                           performs an identity operation.
#     :param input_shape:   Tuple with feature_count, vertical and horizontal size of
#                           input to the layer.
#     :return:
#         CNNLayerDimensions corresponding to identity operation.
#     """
#     dimensions = sd.configuration.CNNLayerDimensions()
#     # No padding
#     dimensions.padding.x = 0
#     dimensions.padding.y = 0
#     # Stride 1
#     dimensions.stride.x = 1
#     dimensions.stride.y = 1
#     # Input shape
#     dimensions.input_shape.feature_count = input_shape[0]
#     dimensions.input_shape.y = input_shape[1]
#     dimensions.input_shape.x = input_shape[2]
#     # Output shape
#     dimensions.output_shape.feature_count = input_shape[0]
#     dimensions.output_shape.y = input_shape[1]
#     dimensions.output_shape.x = input_shape[2]

#     return dimensions


# def identity_weights(feature_count: int) -> List[List[List[List[int]]]]:
#     """
#     identity_weights - Return weights that correspond to identity operation,
#                        assuming that feature_count and channel_count are the same.
#     :param feature_count:  int  Number of input features
#     :return:
#         list    Weights for identity operation
#     """
#     return [
#         [[[int(i == j)]] for j in range(feature_count)] for i in range(feature_count)
#     ]


# def write_to_device(config: Dict, device: samna.SpeckModel, weights=None):
#     """
#     Write your model configuration to dict

#     :param config:
#     :param device:
#     :return:
#     """
#     device.set_config(to_speck_config(config))
#     if weights:
#         device.set_weights(weights)
#     device.apply()


# def to_speck_config(config: Dict) -> samna.SpeckConfig:
#     speck_config = samna.SpeckConfig()
#     # TODO

#     # Populate the config
#     return speck_config
