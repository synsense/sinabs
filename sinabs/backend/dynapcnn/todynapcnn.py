import warnings
from copy import deepcopy
from warnings import warn
import time

from sinabs.backend.dynapcnn.config_factory import ChipFactory

try:
    import samna
except (ImportError, ModuleNotFoundError):
    SAMNA_AVAILABLE = False
else:
    SAMNA_AVAILABLE = True

import torch.nn as nn
import torch
import sinabs.layers as sl
import sinabs
from typing import Tuple, Union, Optional, Sequence, List
from .io import open_device, _parse_device_string, enable_timestamps, disable_timestamps, reset_timestamps
from .dynapcnnlayer import DynapcnnLayer
from .dvslayer import DVSLayer
from .flipdims import FlipDims
from .mapping import get_valid_mapping, dynapcnndevkit_constraints, speck2_constraints
from .utils import convert_model_to_layer_list, build_from_list, infer_input_shape
import json


class DynapcnnCompatibleNetwork(nn.Module):
    """
    Given a sinabs spiking network, prepare a dynapcnn-compatible network.
    This can be used to test the network will be equivalent once on DYNAPCNN.
    This class also provides utilities to make the dynapcnn configuration and
    upload it to DYNAPCNN.

    The following operations are done when converting to dynapcnn-compatible:

    * multiple avg pooling layers in a row are consolidated into one and \
    turned into sum pooling layers;
    * checks are performed on layer hyperparameter compatibility with dynapcnn \
    (kernel sizes, strides, padding)
    * checks are performed on network structure compatibility with dynapcnn \
    (certain layers can only be followed by other layers)
    * linear layers are turned into convolutional layers
    * dropout layers are ignored
    * weights, biases and thresholds are discretized according to dynapcnn requirements
    """

    def __init__(
        self,
        snn: Union[nn.Sequential, sinabs.Network],
        input_shape: Tuple[int, int, int],
        dvs_input: bool = False,
        discretize: bool = True,
    ):
        """
        DynapcnnCompatibleNetwork: a class turning sinabs networks into dynapcnn
        compatible networks, and making dynapcnn configurations.

        Parameters
        ----------
            snn: sinabs.Network
                SNN that determines the structure of the `DynapcnnCompatibleNetwork`
            input_shape: tuple of ints
                Shape of the input, convention: (features, height, width)
            dvs_input: bool
                Does dynapcnn receive input from its DVS camera?
            discretize: bool
                If True, discretize the parameters and thresholds.
                This is needed for uploading weights to dynapcnn. Set to False only for
                testing purposes.
        """
        super().__init__()
        self.chip_layers_ordering = []
        self.compatible_layers = []

        # Convert models  to sequential
        layers = convert_model_to_layer_list(model=snn)
        # Check if dvs input is expected
        if dvs_input or isinstance(layers[0], sl.InputLayer):
            self.dvs_input = True
        else:
            self.dvs_input = False

        input_shape = infer_input_shape(layers, input_shape=input_shape)

        # Build model from layers
        self.sequence = build_from_list(layers, in_shape=input_shape, discretize=discretize)
        # this holds the DynapcnnLayer objects which can be used for testing
        # and also deal with single-layer-level configuration issues
        self.compatible_layers = [*self.sequence]

        # Add a DVS layer in case dvs_input is flagged
        if self.dvs_input and not isinstance(self.compatible_layers[0], DVSLayer):
            dvs_layer = DVSLayer(input_shape=input_shape[1:]) # Ignore the channel dimension
            self.compatible_layers = [dvs_layer] + self.compatible_layers
            self.sequence = nn.Sequential(*self.compatible_layers)

        if self.dvs_input:
            # Enable dvs pixels
            self.compatible_layers[0].disable_pixel_array = False

    def to(
        self,
        device="cpu",
        chip_layers_ordering="auto",
        monitor_layers: Optional[List] = None,
        config_modifier=None,
    ):
        """
        Parameters
        ----------

        device: String
            cpu:0, cuda:0, dynapcnndevkit, speck2devkit

        chip_layers_ordering: List/"auto"
            A list of layers on the device where you want each of the model layers to be placed.

        monitor_layers: None/List
            A list of all chip-layers that you want to monitor.
            If you want to monitor the dvs layer for eg.
            ::

                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8

        config_modifier:
            A user configuration modifier method.
            This function can be used to make any custom changes you want to make to the configuration object.

        Note
        ----
        chip_layers_ordering and monitor_layers are used only when using synsense devices.
        For GPU or CPU usage these options are ignored.
        """
        self.device = device
        if isinstance(device, torch.device):
            return super().to(device)
        elif isinstance(device, str):
            device_name, _ = _parse_device_string(device)
            # TODO: This should probably check with the device type from the factor
            if device_name in ("dynapcnndevkit", "speck2devkit", "speck2b"):
                # Generate config
                config = self.make_config(
                    chip_layers_ordering=chip_layers_ordering,
                    device=device,
                    monitor_layers=monitor_layers,
                    config_modifier=config_modifier,
                )

                # Apply configuration to device
                self.samna_device = open_device(device)

                if device_name == "dynapcnndevkit":
                    self.samna_device.get_model().apply_configuration(config)
                    time.sleep(1)
                    self.samna_output_buffer = (
                        samna.BufferSinkNode_dynapcnn_event_output_event()
                    )
                elif device_name == "speck2devkit":
                    self.samna_device.get_daughter_board(
                        0
                    ).get_model().apply_configuration(config)
                    time.sleep(1)
                    self.samna_output_buffer = (
                        samna.BufferSinkNode_speck2_event_output_event()
                    )
                elif device_name == "speck2b":
                    self.samna_device.get_model().apply_configuration(config)
                    time.sleep(1)
                    self.samna_output_buffer = (
                        samna.BufferSinkNode_speck2b_event_output_event()
                    )
                else:
                    raise ValueError(
                        "Unknown device description. device name has to be dynapcnndekit or speck2devkit"
                    )

                # Connect buffer sink node to device
                self.samna_device.get_model().get_source_node().add_destination(
                    self.samna_output_buffer.get_input_channel()
                )

                return self
            else:
                return super().to(device)
        else:
            raise Exception("Unknown device description.")

    def make_config(
        self,
        chip_layers_ordering: Union[Sequence[int], str] = "auto",
        device="dynapcnndevkit:0",
        monitor_layers: Optional[List] = None,
        config_modifier=None,
    ):
        """
        Prepare and output the `samna` DYNAPCNN configuration for this network.

        Parameters
        ----------

        chip_layers_ordering: sequence of integers or `auto`
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.

        device: String
            dynapcnndevkit:0 or speck2devkit:0

        monitor_layers: None/List
            A list of all chip-layers that you want to monitor.
            If you want to monitor the dvs layer for eg.
            ::

                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8

        config_modifier:
            A user configuration modifier method.
            This function can be used to make any custom changes you want to make to the configuration object.

        Returns
        -------
        Configuration object
            Object defining the configuration for the device

        Raises
        ------
            ImportError
                If samna is not available.
        """

        if not SAMNA_AVAILABLE:
            raise ImportError("`samna` does not appear to be installed.")

        config_builder = ChipFactory(device).get_config_builder()

        # Figure out layer ordering
        if chip_layers_ordering == "auto":
            chip_layers_ordering = config_builder.get_valid_mapping(self)
        else:
            # Truncate chip_layers_ordering just in case a longer list is passed
            chip_layers_ordering = chip_layers_ordering[: len(self.compatible_layers)]

        # Save the chip layers
        self.chip_layers_ordering = chip_layers_ordering
        # Update config
        config = config_builder.build_config(self, chip_layers_ordering)

        # Enable monitors on the specified layers
        # Find layers corresponding to the chip
        if monitor_layers is not None:
            monitor_chip_layers = [self.find_chip_layer(lyr) for lyr in monitor_layers]
            if "dvs" in monitor_layers:
                monitor_chip_layers.append("dvs")
            config_builder.monitor_layers(config, monitor_chip_layers)

        # Fix default factory setting to not return input events (UGLY!! Ideally this should happen in samna)
        # config.factory_settings.monitor_input_enable = False

        # Apply user config modifier
        if config_modifier is not None:
            config = config_modifier(config)

        # Validate config
        if config_builder.validate_configuration(config):
            print("Network is valid")
            return config
        else:
            raise ValueError(f"Generated config is not valid for {device}")

    def find_chip_layer(self, layer_idx):
        """
        Given an index of a layer in the model, find the corresponding chip layer where it is placed

        Parameters
        ----------
        layer_idx: int
            Index of a layer

        Returns
        -------
        chip_lyr_idx: int
            Index of the layer on the chip where the model layer is placed.
        """
        if len(self.chip_layers_ordering) != len(self.compatible_layers):
            raise Exception("Looks like the model has not been mapped onto a device.")

        return self.chip_layers_ordering[layer_idx]

    def _convert_conv2d_layer(
        self,
        layers: Sequence[nn.Module],
        input_shape: Tuple[int],
        rescaling_from_pooling: int,
    ) -> Tuple[int, Tuple[int], int]:
        """
        Generate a DynapcnnLayer from a Conv2d layer and its subsequent spiking and
        pooling layers.

        Parameters
        ----------
            layers: sequence of layer objects
                First object must be Conv2d, next must be a SpikingLayer. All pooling
                layers that follow immediately are consolidated. Layers after this
                will be ignored.
            input_shape: tuple of integers
                Shape of the input to the first layer in `layers`. Convention:
                (input features, height, width)
            rescaling_from_pooling: int
                Weights of Conv2d layer are scaled down by this factor. Can be
                used to account for preceding average pooling that gets converted
                to sum pooling.

        Returns
        -------
            int
                Number of consolidated pooling layers
            tuple
                output shape (features, height, width)
            int
                rescaling factor to account for average pooling

        Raises
        ------
            TypeError
                If `layer[1]` is not of type `SpikingLayer`
        """

        lyr_curr = layers[0]

        # Next layer needs to be spiking
        try:
            lyr_next = layers[1]
        except IndexError:
            raise TypeError(
                "Convolution must be followed by spiking layer, end of network found."
            )
        if not isinstance(lyr_next, sl.iaf_bptt.SpikingLayer):
            raise TypeError(
                f"Convolution must be followed by spiking layer, found {type(lyr_next)}"
            )

        # - Consolidate pooling from subsequent layers
        pooling: Union[List[int], int]
        pooling, i_next, rescaling = consolidate_pooling(
            layers[2:], dvs=False, discretize=self._discretize
        )

        # The DynapcnnLayer object knows how to turn the conv-spk-pool trio to
        # a dynapcnn layer, and has a forward method, and computes the output shape
        compatible_object = DynapcnnLayer(
            conv=lyr_curr,
            spk=lyr_next,
            pool=pooling,
            in_shape=input_shape,
            discretize=self._discretize,
            rescale_weights=rescaling_from_pooling,
        )
        # the previous rescaling has been used, the new one is used in the next layer
        rescaling_from_pooling = rescaling
        # we save this object for future forward passes for testing
        self.compatible_layers.append(compatible_object)
        output_shape = compatible_object.output_shape

        return i_next, output_shape, rescaling_from_pooling

    def forward(self, x):
        if hasattr(self, "device") and _parse_device_string(self.device)[0] in (
            "dynapcnndevkit",
            "speck2devkit",
            "speck2b",
        ):
            _ = self.samna_output_buffer.get_events()  # Flush buffer
            # NOTE: The code to start and stop time stamping is device specific
            reset_timestamps(self.device)
            enable_timestamps(self.device)
            # Send input
            self.samna_device.get_model().write(x)
            time.sleep((x[-1].timestamp - x[0].timestamp) * 1e-6 + 1)
            # Disable timestamp
            disable_timestamps(self.device)
            # Read events back
            evsOut = self.samna_output_buffer.get_events()
            return evsOut
        else:
            """Torch's forward pass."""
            self.eval()
            with torch.no_grad():
                return self.sequence(x)

    def memory_summary(self):
        """
        Get a summary of the network's memory requirements

        Returns
        -------
        dict:
            A dictionary with keys kernel, neuron, bias.
            The values are a list of the corresponding number per layer in the same order as the model

        """
        summary = {}
        summary.update({k: list() for k in self.sequence[0].memory_summary().keys()})
        for lyr in self.sequence:
            lyr_summary = lyr.memory_summary()
            for k, v in lyr_summary.items():
                summary[k].append(v)
        return summary


def enable_monitors(config, monitor_chip_layers: List):
    """
    Updates the config object in place.

    Parameters
    ----------
    config:
        samna config object
    monitor_chip_layers:
        The layers to be monitored on the chip.

    Returns
    -------
    config:
        Returns the modified config. (The config object is modified in place)
    """
    warnings.warn("Use ConfigBuilder.monitor_layers instead")
    monitor_layers = monitor_chip_layers.copy()
    if "dvs" in monitor_layers:
        config.dvs_layer.monitor_enable = True
        config.dvs_layer.monitor_sensor_enable = True
        monitor_layers.remove("dvs")
    for lyr_indx in monitor_layers:
        config.cnn_layers[lyr_indx].monitor_enable = True
    return config


def consolidate_pooling(
    layers: Sequence[nn.Module], dvs: bool, discretize: bool
) -> Tuple[Union[List[int], int], int, int]:
    """
    Consolidate the first `AvgPool2d` objects in `layers` until the first object of different type.

    Parameters
    ----------
        layers: Sequence of layer objects
            Contains `AvgPool2d` and other objects.
        dvs: bool
            If `True`, x- and y- pooling may be different and a tuple is returned instead of an integer.
        discritize: bool
            True if the weights of the model need to be discretized

    Returns
    -------
        int or tuple of ints
            Consolidated pooling size. Tuple if `dvs` is `True`.
        int or None
            Index of first object in `layers` that is not a `AvgPool2d`,
            or `None`, if all objects in `layers` are `AvgPool2d`.
        int or None
            Rescaling factor needed when turning AvgPool to SumPool. May
            differ from the pooling kernel in certain cases.
    """
    pooling = [1, 1] if dvs else 1
    rescaling_factor = 1

    for i_next, lyr in enumerate(layers):
        if isinstance(lyr, nn.AvgPool2d):
            if discretize:
                warn(
                    "For average pooling, subsequent weights are scaled down. This can lead to larger quantization errors when `discretize` is `True`."
                )
            # Update pooling size
            new_pooling = get_pooling_size(lyr, dvs=dvs)
            if dvs:
                pooling[0] *= new_pooling[0]
                pooling[1] *= new_pooling[1]
                rescaling_factor *= new_pooling[0] * new_pooling[1]
            else:
                pooling *= new_pooling
                rescaling_factor *= new_pooling ** 2
        elif isinstance(lyr, sl.SumPool2d):
            # Update pooling size
            new_pooling = get_pooling_size(lyr, dvs=dvs, check_pad=False)
            if dvs:
                pooling[0] *= new_pooling[0]
                pooling[1] *= new_pooling[1]
            else:
                pooling *= new_pooling
        else:
            # print("Pooling:", pooling)
            # print("Output shape:", input_shape)
            return pooling, i_next, rescaling_factor

    # If this line is reached, all objects in `layers` are pooling layers.
    return pooling, None, rescaling_factor


def get_pooling_size(
    layer: nn.AvgPool2d, dvs: bool, check_pad: bool = True
) -> Union[int, Tuple[int]]:
    """
    Determine the pooling size of a pooling object.

    Parameters
    ----------
        layer: torch.nn.AvgPool2d)
            Pooling layer
        dvs: bool
            If `True`, pooling does not need to be symmetric.
        check_pad: bool
            If `True` (default), check that padding is zero.

    Returns
    -------
        int or tuple of int
            Pooling size. If `dvs` is `True`, return a tuple with sizes for
            y- and x-pooling.
    """

    # Warn if there is non-zero padding.
    # Padding can be either int or tuple of ints
    if check_pad:
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
    pooling_y, pooling_x = (pooling, pooling) if isinstance(pooling, int) else pooling

    stride = layer.stride
    stride_y, stride_x = (stride, stride) if isinstance(stride, int) else stride

    if dvs:
        # Check whether pooling and strides match
        if stride_y != pooling_y or stride_x != pooling_x:
            raise ValueError("AvgPool2d: Stride size must be the same as pooling size.")
        return (pooling_y, pooling_x)
    else:
        # Check whether pooling is symmetric
        if pooling_x != pooling_y:
            raise ValueError("AvgPool2d: Pooling must be symmetric for CNN layers.")
        pooling = pooling_x  # Is this the vertical dimension?
        # Check whether pooling and strides match
        if any(stride != pooling for stride in (stride_x, stride_y)):
            raise ValueError("AvgPool2d: Stride size must be the same as pooling size.")
        return pooling


def _merge_conv_bn(conv, bn):
    """
    Merge a convolutional layer with subsequent batch normalization

    Parameters
    ----------
        conv: torch.nn.Conv2d
            Convolutional layer
        bn: torch.nn.Batchnorm2d
            Batch normalization

    Returns
    -------
        torch.nn.Conv2d: Convolutional layer including batch normalization
    """
    mu = bn.running_mean
    sigmasq = bn.running_var

    if bn.affine:
        gamma, beta = bn.weight, bn.bias
    else:
        gamma, beta = 1.0, 0.0

    factor = gamma / sigmasq.sqrt()

    c_weight = conv.weight.data.clone().detach()
    c_bias = 0.0 if conv.bias is None else conv.bias.data.clone().detach()

    conv = deepcopy(conv)  # TODO: this will cause copying twice

    conv.weight.data = c_weight * factor[:, None, None, None]
    conv.bias.data = beta + (c_bias - mu) * factor

    return conv


def write_dynapcnn_layer_config(config_dict: dict, chip_layer: "CNNLayerConfig"):
    """
    Write a single layer configuration to the dynapcnn conf object.

    Parameters
    ----------
        config_dict: dict
            Dict containing the configuration
        chip_layer: CNNLayerConfig
            DYNAPCNN configuration object representing the layer to which
            configuration is written.
    """
    warnings.warn("Use ConfigBuilder.write_dynapcnn_layer_config() instead")
    # Update configuration of the DYNAPCNN layer
    chip_layer.dimensions = config_dict["dimensions"]
    for i in range(len(config_dict["destinations"])):
        if "pooling" in config_dict["destinations"][i]:
            chip_layer.destinations[i].pooling = config_dict["destinations"][i]["pooling"]

    for param, value in config_dict.items():
        if (param != "dimensions") and (param != "destinations"):
            try:
                setattr(chip_layer, param, value)
            except TypeError as e:
                raise TypeError(f"Unexpected parameter {param} or value. {e}")


def write_dvs_layer_config(config_dict: dict, config: "DvsLayerConfig"):
    """
    Update config of the dvs layer

    Parameters
    ----------
    config

    """
    warnings.warn("Use ConfigBuilder.write_dvs_layer_config() instead")
    print(config_dict)
    # Apply to config
    for param, value in config_dict.items():
        setattr(config, param, value)


def write_model_to_config(model: nn.Sequential, config, chip_layers: Sequence[int]):
    warnings.warn("Use ConfigBuilder.build_config() instead")
    i_layer_chip = 0
    for i, chip_equivalent_layer in enumerate(model):
        if isinstance(chip_equivalent_layer, DVSLayer):
            chip_layer = config.dvs_layer
            write_dvs_layer_config(chip_equivalent_layer.get_config_dict(), chip_layer)
        elif isinstance(chip_equivalent_layer, DynapcnnLayer):
            # Object representing DYNAPCNN layer
            chip_layer = config.cnn_layers[chip_layers[i_layer_chip]]
            # read the configuration dictionary from DynapcnnLayer
            # and write it to the dynapcnn configuration object
            write_dynapcnn_layer_config(chip_equivalent_layer.get_config_dict(), chip_layer)

        elif isinstance(chip_equivalent_layer, sl.InputLayer):
            pass
        else:
            # in our generated network there is a spurious layer...
            # should never happen
            raise TypeError("Unexpected layer in generated network")

        if i == len(model) - 1:
            # last layer
            chip_layer.destinations[0].enable = False
        else:
            i_layer_chip += 1
            # Set destination layer
            chip_layer.destinations[0].layer = chip_layers[i_layer_chip]
            chip_layer.destinations[0].enable = True


def validate_configuration(config, device: str) -> bool:
    """
    Verify whether the config object is valid given a device's specifications
    Parameters
    ----------
    config: Config object
        Config object of a device
    device: String
        dynapcnndevkit or speck2devkit

    Returns
    -------
    true if valid

    """
    warnings.warn("Use ConfigBuilder.validate_configuration() instead")
    # Validate configuration
    if device == "dynapcnndevkit":
        is_valid, message = samna.dynapcnn.validate_configuration(config)
    elif device == "speck2devkit":
        is_valid, message = samna.speck2.validate_configuration(config)
    else:
        raise Exception(f"Unknown device type {device}")
    if not is_valid:
        print(message)
    return is_valid
