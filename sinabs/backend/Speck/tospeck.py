from warnings import warn

import torch.nn as nn
import sinabs.layers as sl
from sinabs.cnnutils import infer_output_shape
from typing import Dict, Tuple, Union, Optional

import samna.speck as sd

# import speckdemo as sd
from .SpeckLayer import SpeckLayer


class SpeckCompatibleNetwork(nn.Module):
    def __init__(
        self,
        snn: Union[nn.Module, sl.TorchLayer],
        input_shape: Optional[Tuple[int]] = None,
        dvs_input: bool = True,
    ) -> Dict:
        """
        Build a configuration object of a given module

        :param snn: sinabs.Network or sinabs.layers.TorchLayer instance
        """
        super().__init__()

        self.config = sd.configuration.SpeckConfiguration()

        self.compatible_layers = []

        # TODO: Currently only spiking seq. models are supported
        try:
            layers = list(snn.spiking_model.seq)
        except AttributeError:
            raise ValueError("`snn` must contain a sequential spiking model.")

        i_layer = 0
        i_layer_speck = 0

        # - Input to start with
        if isinstance(layers[0], sl.InputLayer):
            input_shape = layers[0].output_shape
            i_layer += 1
        elif input_shape is None:
            raise ValueError(
                "`input_shape` must be provided if first layer is not `InputLayer`."
            )
        if dvs_input:
            # - Cut DVS output to match output shape of `lyr_curr`
            dvs = self.config.dvs_layer
            dvs.cut.y = input_shape[1]
            dvs.cut.x = input_shape[2]
            # TODO: How to deal with feature count?

        # - Iterate over layers from model
        while i_layer < len(layers):

            # Layer to be ported to Speck
            lyr_curr = layers[i_layer]

            if isinstance(lyr_curr, (nn.Conv2d, nn.Linear)):
                # Object representing Speck layer
                i_next, input_shape = self._handle_conv2d_layer(
                    layers[i_layer:], input_shape, i_layer_speck,
                )

                if i_next is None:
                    # TODO: How to route to readout layer? Does destination need to be set?
                    break
                else:
                    # Add 2 to i_layer to go to next layer, + i_next for number
                    # of consolidated pooling layers
                    i_layer += i_next + 2
                    i_layer_speck += 1

            elif isinstance(lyr_curr, nn.AvgPool2d):
                # This case can only happen when `layers` starts with a pooling layer, or
                # input layer because all other pooling layers should get consolidated.
                # Assume that input comes from DVS.

                # Object representing Speck DVS
                dvs = self.config.dvs_layer
                pooling, i_next = self.consolidate_pooling(layers[i_layer:], dvs=True)
                dvs.pooling.y, dvs.pooling.x = pooling
                self.compatible_layers.append(
                    nn.AvgPool2d(kernel_size=pooling, stride=pooling)
                )

                if i_next is not None:
                    dvs.destinations[0].layer = i_layer_speck
                    dvs.destinations[0].enable = True
                    i_layer += i_next
                else:
                    break

            elif isinstance(lyr_curr, nn.Dropout2d):
                # - Ignore dropout layers
                i_layer += 1

            elif isinstance(lyr_curr, nn.Flatten):
                # input_shape = infer_output_shape(lyr_curr, input_shape)
                # if len(input_shape) < 3:
                #     # - Fill shape with 1's to match expected dimensions
                #     input_shape += (3 - len(input_shape)) * (1,)
                i_layer += 1

                # if i_layer == len(layers):
                #     raise TypeError("Final layer cannot be of type `Flatten`")

            elif isinstance(lyr_curr, sl.InputLayer):
                raise TypeError(f"Only first layer can be of type {type(lyr_curr)}.")

            elif isinstance(lyr_curr, sl.iaf_bptt.SpikingLayer):
                raise TypeError(
                    "`SpikingLayer` must be preceded by layer of type `Conv2d` or `Linear`."
                )

            else:
                raise TypeError(
                    f"Layers of type {type(lyr_curr)} are currently not supported."
                )

        # TODO: Does anything need to be done after iterating over layers?

        # print("Finished configuration of Speck.")

        self.sequence = nn.Sequential(*self.compatible_layers)

    def _handle_conv2d_layer(self, layers, input_shape, i_layer_speck):

        lyr_curr = layers[0]

        # Object representing Speck layer
        speck_layer = self.config.cnn_layers[i_layer_speck]

        # Next layer needs to be spiking
        try:
            lyr_next = layers[1]
        except IndexError:
            reached_end = True
        else:
            reached_end = False
        if reached_end or not isinstance(lyr_next, sl.iaf_bptt.SpikingLayer):
            raise TypeError("Convolutional layer must be followed by spiking layer.")

        # Extract configuration specs from layer objects and update Speck config
        # input_shape = self.spiking_conv2d_to_speck(
        #     input_shape, lyr_curr, lyr_next, speck_layer
        # )

        # - Consolidate pooling from subsequent layers
        pooling, _, i_next = self.consolidate_pooling(layers[2:], input_shape)

        compatible_object = SpeckLayer(conv=lyr_curr, spk=lyr_next, pool=pooling, in_shape=input_shape)
        self.compatible_layers.append(compatible_object)
        input_shape = self.write_speck_config(input_shape, compatible_object.config_dict, speck_layer)

        # For now: Sequential model, second destination always disabled
        speck_layer.destinations[1].enable = False

        if i_next is not None:
            # Set destination layer
            speck_layer.destinations[0].layer = i_layer_speck + 1
            speck_layer.destinations[0].pooling = pooling
            speck_layer.destinations[0].enable = True

        else:
            speck_layer.destinations[0].enable = False

        return i_next, input_shape

    def forward(self, x):
        for l in self.sequence:
            print(x.shape)
            x = l(x)
        return x

    def write_speck_config(
        self,
        input_shape: dict,
        config_dict: dict,
        speck_layer,  #: sd.configuration.CNNLayerConfig,
    ) -> Tuple[int]:
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

        # Output shape with given input
        dimensions = config_dict["dimensions"]
        output_shape = (
            dimensions["output_feature_count"],
            dimensions["output_size_y"],
            dimensions["output_size_x"],
        )
        print("Output shape:", output_shape)
        return output_shape

    def consolidate_pooling(
        self, layers, input_shape: Tuple[int], dvs: bool = False
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

        pooling = (1, 1) if dvs else 1

        for i_next, lyr in enumerate(layers):
            if isinstance(lyr, nn.AvgPool2d):
                # Update pooling size
                new_pooling, input_shape = self.get_pooling_size(lyr, input_shape, dvs=dvs)
                if dvs:
                    pooling[0] *= new_pooling[0]
                    pooling[1] *= new_pooling[1]
                else:
                    pooling *= new_pooling
            else:
                # print("Pooling:", pooling)
                # print("Output shape:", input_shape)
                return pooling, input_shape, i_next

        # If this line is reached, all objects in `layers` are `SumPooling2dLayer`s.
        # print("Pooling:", pooling)
        # print("Output shape:", input_shape)
        return pooling, input_shape, None

    def get_pooling_size(
        self, layer: nn.AvgPool2d, input_shape: Tuple[int], dvs: bool = True
    ) -> Union[int, Tuple[int]]:
        """
        get_sumpool2d_pooling_size - Determine the pooling size of a `SumPooling2dLayer` object.
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
            return (pooling_y, pooling_x), infer_output_shape(layer, input_shape)
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
            # TODO: infer_output_shape does not work with discretized
            return pooling, infer_output_shape(layer, input_shape)


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
