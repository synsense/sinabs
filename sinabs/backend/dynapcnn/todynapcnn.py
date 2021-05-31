from copy import deepcopy
from warnings import warn
import time
try:
    import samna
except (ImportError, ModuleNotFoundError):
    SAMNA_AVAILABLE = False
else:
    SAMNA_AVAILABLE = True

import numpy as np
from .dynapcnnlayer import DynapcnnLayer
from .valid_mapping import get_valid_mapping
import torch.nn as nn
import torch
import sinabs.layers as sl
import sinabs
from typing import Tuple, Union, Optional, Sequence, List
from .io import open_device, _parse_device_string, enable_timestamps, disable_timestamps


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
            input_shape: Optional[Tuple[int, int, int]] = None,
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

        # this holds the DynapcnnLayer objects which can be used for testing
        # and also deal with single-layer-level configuration issues
        self.compatible_layers = []

        # TODO: Currently only spiking seq. models are supported
        if isinstance(snn, sinabs.Network):
            submodules = list(snn.spiking_model.children())
            if len(submodules) != 1:
                raise ValueError("Found multiple submodules instead of sequential")
            layers = [*submodules[0]]
        elif isinstance(snn, nn.Sequential):
            layers = [*snn]
        else:
            raise TypeError("Expected torch.nn.Sequential or sinabs.Network")

        # index that goes over the layers of the input network
        i_layer = 0
        # used to carry pooling info to next conv, to rescale weights due to
        # the difference between sum and average pooling
        rescaling_from_pooling = 1

        # - Input to start with
        if isinstance(layers[0], sl.InputLayer):
            input_layer = deepcopy(layers[0])
            if input_shape is not None and input_shape != input_layer.input_shape:
                warn(
                    "Network starts with `InputLayer`. Will ignore `input_shape` argument."
                )
            input_shape = input_layer.input_shape
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
            # Layer to be ported to DYNAPCNN
            lyr_curr = layers[i_layer]

            if isinstance(lyr_curr, (nn.Conv2d, nn.Linear)):
                # Check for batchnorm after conv
                if len(layers) > i_layer + 1:
                    if isinstance(layers[i_layer + 1], nn.BatchNorm2d):
                        lyr_curr = _merge_conv_bn(lyr_curr, layers[i_layer + 1])
                        i_layer += 1

                # Linear and Conv layers are dealt with in the same way.
                i_next, input_shape, rescaling_from_pooling = self._handle_conv2d_layer(
                    [lyr_curr] + layers[i_layer + 1:],
                    input_shape,
                    rescaling_from_pooling,
                )

                if i_next is None:
                    # TODO: How to route to readout layer? Does destination need to be set?
                    break
                else:
                    # Add 2 to i_layer to go to next layer, + i_next for number
                    # of consolidated pooling layers
                    i_layer += i_next + 2

            elif isinstance(lyr_curr, (sl.SumPool2d, nn.AvgPool2d)):
                # This case can only happen if `self.sequence` starts with a pooling layer
                # or input layer because all other pooling layers should get consolidated.
                # Therefore, require that input comes from DVS.
                if not dvs_input:
                    raise TypeError(
                        "First layer cannot be pooling if `dvs_input` is `False`."
                    )
                pooling, i_next, rescaling_from_pooling = consolidate_pooling(
                    layers[i_layer:], dvs=True, discretize=self._discretize
                )

                input_shape = [
                    input_shape[0],
                    input_shape[1] // pooling[0],
                    input_shape[2] // pooling[1],
                ]

                self.compatible_layers.append(
                    sl.SumPool2d(kernel_size=pooling, stride=pooling)
                )

                # if isinstance(lyr_curr, nn.AvgPool2d):
                #     rescaling_from_pooling = pooling[0] * pooling[1]

                if i_next is not None:
                    i_layer += i_next
                else:
                    break

            elif isinstance(lyr_curr, (nn.Dropout, nn.Dropout2d, nn.Flatten)):
                # - Ignore dropout and flatten layers
                i_layer += 1

            else:
                raise TypeError(
                    f"Layers of type {type(lyr_curr)} are not supported here."
                )

        # TODO: Does anything need to be done after iterating over layers?
        # print("Finished configuration of DYNAPCNN.")

        if rescaling_from_pooling != 1:
            warn(
                "Average pooling layer at the end of the network could not "
                "be turned into sum pooling. The output will be different by "
                f"a factor of {rescaling_from_pooling}!"
            )

        self.sequence = nn.Sequential(*self.compatible_layers)

    def to(self, device="cpu", chip_layers_ordering="auto", monitor_layers: List=None, config_modifier=None):
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
                ``
                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                ``
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
            if device_name in ("dynapcnndevkit", "speck2devkit"):
                # Generate config
                config = self.make_config(chip_layers_ordering=chip_layers_ordering, device=device)

                # Enable monitors
                if monitor_layers is not None:
                    monitor_layers = monitor_layers.copy()
                    if "dvs" in monitor_layers:
                        config.dvs_layer.monitor_enable = True
                        config.dvs_layer.monitor_sensor_enable = True
                        monitor_layers.remove("dvs")
                    for lyr_indx in monitor_layers:
                        config.cnn_layers[lyr_indx].monitor_enable = True

                # Apply user config modifier
                if config_modifier is not None:
                    config = config_modifier(config)

                # Apply configuration to device
                self.samna_device = open_device(device)

                if device_name == "dynapcnndevkit":
                    self.samna_device.get_model().apply_configuration(config)
                    time.sleep(1)
                    self.samna_output_buffer = samna.BufferSinkNode_dynapcnn_event_output_event()
                elif device_name == "speck2devkit":
                    self.samna_device.get_daughter_board(0).get_model().apply_configuration(config)
                    time.sleep(1)
                    self.samna_output_buffer = samna.BufferSinkNode_speck2_event_output_event()
                else:
                    raise ValueError("Unknown device description. device name has to be dynapcnndekit or speck2devkit")

                # Connect buffer sink node to device
                self.samna_device.get_model().get_source_node().add_destination(
                    self.samna_output_buffer.get_input_channel()
                )

                return self
            else:
                return super().to(device)
        else:
            raise Exception("Unknown device description.")

    def make_config(self, chip_layers_ordering: Union[Sequence[int], str] = range(9), device="dynapcnndevkit:0"):
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

        Parameters
        ----------
            chip_layers_ordering: sequence of integers or "auto"
                The order in which the dynapcnn layers will be used. If "auto",
                an automated procedure will be used to find a valid ordering.
            device: String
                dynapcnndevkit:0 or speck2devkit:0

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

        if chip_layers_ordering == "auto":
            chip_layers = range(9)  # start with default, correct if necessary
        else:
            chip_layers = chip_layers_ordering

        device_name, _ = _parse_device_string(device)

        if device_name == "dynapcnndevkit":
            config = samna.dynapcnn.configuration.DynapcnnConfiguration()
        elif device_name == "speck2devkit":
            config = samna.speck2.configuration.SpeckConfiguration()
        else:
            raise Exception(f"Unknown device type {device_name}")

        i_layer_chip = 0
        dvs = config.dvs_layer
        if self._dvs_input:
            if self._external_input_shape[0] == 1:
                dvs.merge = True
            elif self._external_input_shape[0] != 2:
                message = "dvs layer must have 1 or 2 input channels"
                raise ValueError("Network not valid for DYNAPCNN\n" + message)

            # - Cut DVS output to match output shape of `lyr_curr`
            dvs.cut.y = self._external_input_shape[1] - 1
            dvs.cut.x = self._external_input_shape[2] - 1
            # - Set DVS destination
            dvs.destinations[0].enable = True
            dvs.destinations[0].layer = chip_layers[i_layer_chip]
            # - Pooling will only be set to > 1 later if applicable
            dvs.pooling.y, dvs.pooling.x = 1, 1


        else:
            dvs.destinations[0].enable = False
        # TODO: Modify in case of non-sequential models
        dvs.destinations[1].enable = False

        ## Update config object according to model specifications
        # Check for first layer to be sumpool, while ignoring InputLayer
        for first_layer in self.sequence:
            if isinstance(first_layer, sl.InputLayer):
                pass
            else:
                break
        if isinstance(first_layer, sl.SumPool2d):
            if self._dvs_input:
                assert (
                        first_layer.stride == first_layer.kernel_size
                )
                dvs.pooling.y, dvs.pooling.x = first_layer.kernel_size
            else:
                raise ValueError("Network cannot start with pooling if dvs_input=False")

        write_model_to_config(self.sequence, config, chip_layers)

        # Validate config
        if validate_configuration(config, device_name):
            print("Network is valid")
            return config
        else:
            if chip_layers_ordering is "auto":
                # Try to auto arrange the layers
                mapping = get_valid_mapping(config)
                if mapping == []:
                    raise ValueError("Could not find valid layer sequence for this network")

                # turn the mapping into a dict
                mapping = {m[0]: m[1] for m in mapping}
                # apply the mapping
                ordering = [mapping[i] for i in chip_layers]

                print("Not valid, trying ordering", ordering)
                return self.make_config(chip_layers_ordering=ordering, device=device)
            else:
                raise ValueError(f"Network not valid for {device}")

    def _handle_conv2d_layer(
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
        pooling, i_next, rescaling = consolidate_pooling(layers[2:], dvs=False, discretize=self._discretize)

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
        if hasattr(self, "device") and _parse_device_string(self.device)[0] in ("dynapcnndevkit", "speck2devkit"):
            _ = self.samna_output_buffer.get_events()  # Flush buffer
            # NOTE: The code to start and stop time stamping is device specific
            disable_timestamps(self.device)
            enable_timestamps(self.device)
            # Send input
            self.samna_device.get_model().write(x)
            time.sleep((x[-1].timestamp - x[0].timestamp)*1e-6 + 1)
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


def consolidate_pooling(layers: Sequence[nn.Module], dvs: bool, discretize: bool) -> Tuple[Union[List[int], int], int, int]:
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


def get_pooling_size(layer: nn.AvgPool2d, dvs: bool, check_pad: bool = True) -> Union[int, Tuple[int]]:
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
    pooling_y, pooling_x = (
        (pooling, pooling) if isinstance(pooling, int) else pooling
    )

    stride = layer.stride
    stride_y, stride_x = (stride, stride) if isinstance(stride, int) else stride

    if dvs:
        # Check whether pooling and strides match
        if stride_y != pooling_y or stride_x != pooling_x:
            raise ValueError(
                "AvgPool2d: Stride size must be the same as pooling size."
            )
        return (pooling_y, pooling_x)
    else:
        # Check whether pooling is symmetric
        if pooling_x != pooling_y:
            raise ValueError("AvgPool2d: Pooling must be symmetric for CNN layers.")
        pooling = pooling_x  # Is this the vertical dimension?
        # Check whether pooling and strides match
        if any(stride != pooling for stride in (stride_x, stride_y)):
            raise ValueError(
                "AvgPool2d: Stride size must be the same as pooling size."
            )
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

    # Update configuration of the DYNAPCNN layer
    chip_layer.dimensions = config_dict["dimensions"]

    chip_layer.weights = config_dict["weights"]
    chip_layer.biases = config_dict["biases"]
    chip_layer.weights_kill_bit = config_dict["weights_kill_bit"]
    chip_layer.biases_kill_bit = config_dict["biases_kill_bit"]
    chip_layer.neurons_initial_value = config_dict["neurons_state"]
    chip_layer.neurons_value_kill_bit = config_dict["neurons_state_kill_bit"]
    chip_layer.leak_enable = config_dict["leak_enable"]

    for param, value in config_dict["layer_params"].items():
        # print(f"Setting parameter {param}: {value}")
        setattr(chip_layer, param, value)


def write_model_to_config(model: nn.Sequential, config, chip_layers: Sequence[int]):
    i_layer_chip = 0
    for i, chip_equivalent_layer in enumerate(model):
        # happens when the network starts with pooling
        if isinstance(chip_equivalent_layer, sl.SumPool2d):
            pass
        elif isinstance(chip_equivalent_layer, DynapcnnLayer):
            # Object representing DYNAPCNN layer
            chip_layer = config.cnn_layers[chip_layers[i_layer_chip]]
            # read the configuration dictionary from DynapcnnLayer
            # and write it to the dynapcnn configuration object
            write_dynapcnn_layer_config(chip_equivalent_layer.config_dict, chip_layer)

            # For now: Sequential model, second destination always disabled
            chip_layer.destinations[1].enable = False

            if i == len(model) - 1:
                # last layer
                chip_layer.destinations[0].enable = False
            else:
                i_layer_chip += 1
                # Set destination layer
                chip_layer.destinations[0].layer = chip_layers[i_layer_chip]
                chip_layer.destinations[
                    0
                ].pooling = chip_equivalent_layer.config_dict["Pooling"]
                chip_layer.destinations[0].enable = True
        elif isinstance(chip_equivalent_layer, sl.InputLayer):
            pass
        else:
            # in our generated network there is a spurious layer...
            # should never happen
            raise TypeError("Unexpected layer in generated network")


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
    # Validate configuration
    if device == "dynapcnndevkit":
        is_valid, message = samna.dynapcnn.validate_configuration(config)
    elif device == "speck2devkit":
        is_valid, message = samna.speck2.validate_configuration(config)
    else:
        raise Exception(f"Unknown device type {device}")
    return is_valid

