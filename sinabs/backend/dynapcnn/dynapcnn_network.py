import time
from subprocess import CalledProcessError
from sinabs.backend.dynapcnn.chip_factory import ChipFactory
from .exceptions import InputConfigurationError

try:
    import samna
except (ImportError, ModuleNotFoundError, CalledProcessError):
    SAMNA_AVAILABLE = False
else:
    # IO module only works if samna is available
    from .io import (
        open_device,
        enable_timestamps,
        disable_timestamps,
        reset_timestamps,
    )

    SAMNA_AVAILABLE = True

import torch
import torch.nn as nn
import sinabs
from typing import Tuple, Union, Optional, Sequence, List
from .dynapcnn_layer import DynapcnnLayer
from .dvs_layer import DVSLayer
from .utils import convert_model_to_layer_list, build_from_list, infer_input_shape, _parse_device_string


class DynapcnnNetwork(nn.Module):
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
        DynapcnnNetwork: a class turning sinabs networks into dynapcnn
        compatible networks, and making dynapcnn configurations.

        Parameters
        ----------
            snn: sinabs.Network
                SNN that determines the structure of the `DynapcnnNetwork`
            input_shape: None or tuple of ints
                Shape of the input, convention: (features, height, width)
                If None, `snn` needs an InputLayer
            dvs_input: bool
                Does dynapcnn receive input from its DVS camera?
            discretize: bool
                If True, discretize the parameters and thresholds.
                This is needed for uploading weights to dynapcnn. Set to False only for
                testing purposes.
        """
        super().__init__()

        # This attribute stores the location/core-id of each of the DynapcnnLayers upon placement on chip
        self.chip_layers_ordering = []

        self.compatible_layers = []
        self.input_shape = input_shape
        # Convert models  to sequential
        layers = convert_model_to_layer_list(model=snn)
        # Check if dvs input is expected
        if dvs_input:
            self.dvs_input = True
        else:
            self.dvs_input = False

        input_shape = infer_input_shape(layers, input_shape=input_shape)

        if len(input_shape) != 3:
            raise InputConfigurationError(
                f"input_shape expected to have length 3 or None but input_shape={input_shape} given.")

        # Build model from layers
        self.sequence = build_from_list(
            layers, in_shape=input_shape, discretize=discretize
        )
        # this holds the DynapcnnLayer objects which can be used for testing
        # and also deal with single-layer-level configuration issues
        self.compatible_layers = [*self.sequence]

        # Add a DVS layer in case dvs_input is flagged
        if self.dvs_input and not isinstance(self.compatible_layers[0], DVSLayer):
            dvs_layer = DVSLayer(
                input_shape=input_shape[1:]
            )  # Ignore the channel dimension
            self.compatible_layers = [dvs_layer] + self.compatible_layers
            self.sequence = nn.Sequential(*self.compatible_layers)
        

        if self.dvs_input:
            # Enable dvs pixels
            self.compatible_layers[0].disable_pixel_array = False

    def to(
        self,
        device="cpu",
        chip_layers_ordering="auto",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
    ):
        """
        Parameters
        ----------

        device: String
            cpu:0, cuda:0, dynapcnndevkit, speck2devkit

        chip_layers_ordering: sequence of integers or `auto`
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            Note: This list should be the same length as the number of dynapcnn layers in your model.

        monitor_layers: None/List
            A list of all chip-layers that you want to monitor.
            If you want to monitor the dvs layer for eg.
            ::

                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                monitor_layers = "all" # If you want to monitor all the layers

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
            if device_name in ChipFactory.supported_devices:
                # Generate config
                config = self.make_config(
                    chip_layers_ordering=chip_layers_ordering,
                    device=device,
                    monitor_layers=monitor_layers,
                    config_modifier=config_modifier,
                )

                # Apply configuration to device
                self.samna_device = open_device(device)
                self.samna_device.get_model().apply_configuration(config)
                time.sleep(1)

                builder = ChipFactory(device).get_config_builder()
                # Create input source node
                self.samna_input_buffer = builder.get_input_buffer()
                # Create output sink node node 
                self.samna_output_buffer = builder.get_output_buffer()

                # Connect source node to device sink
                self.device_input_graph = samna.graph.EventFilterGraph()
                self.device_input_graph.sequential(
                    [
                        self.samna_input_buffer,
                        self.samna_device.get_model().get_sink_node()
                    ]
                )

                # Connect sink node to device
                self.device_output_graph = samna.graph.EventFilterGraph()
                self.device_output_graph.sequential(
                    [
                        self.samna_device.get_model().get_source_node(),
                        self.samna_output_buffer
                    ]
                )
                self.device_input_graph.start()
                self.device_output_graph.start()
                self.samna_config = config
                return self
            else:
                return super().to(device)
        else:
            raise Exception("Unknown device description.")

    def make_config(
        self,
        chip_layers_ordering: Union[Sequence[int], str] = "auto",
        device="dynapcnndevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
    ):
        """
        Prepare and output the `samna` DYNAPCNN configuration for this network.

        Parameters
        ----------

        chip_layers_ordering: sequence of integers or `auto`
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            Note: This list should be the same length as the number of dynapcnn layers in your model.

        device: String
            dynapcnndevkit, speck2b or speck2devkit

        monitor_layers: None/List/Str
            A list of all chip-layers that you want to monitor.
            If you want to monitor the dvs layer for eg.
            ::

                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                monitor_layers = "all" # If you want to monitor all the layers

            If this value is left as None, by default the last layer of the model is monitored.

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
            if self.dvs_input:
                chip_layers_ordering = chip_layers_ordering[: len(self.compatible_layers)-1]
            chip_layers_ordering = chip_layers_ordering[: len(self.compatible_layers)]

        # Save the chip layers
        self.chip_layers_ordering = chip_layers_ordering
        # Update config
        config = config_builder.build_config(self, chip_layers_ordering)
        if self.input_shape and self.input_shape[0] == 1:
            config.dvs_layer.merge = True
        # Check if any monitoring is enabled and if not, enable monitoring for the last layer
        if monitor_layers is None:
            monitor_layers = [-1]
        elif monitor_layers == 'all':
            monitor_layers = list(range(len(self.compatible_layers)))

        # Enable monitors on the specified layers
        # Find layers corresponding to the chip
        monitor_chip_layers = [self.find_chip_layer(lyr) for lyr in monitor_layers if lyr != "dvs"]
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

    def reset_states(self, randomize=False):
        """
        Reset the states of the network.
        """
        if hasattr(self, "device") and isinstance(self.device, str):
            device_name, _ = _parse_device_string(self.device)
            if device_name in ChipFactory.supported_devices:
                config_builder = ChipFactory(self.device).get_config_builder()
                # Set all the vmem states in the samna config to zero
                config_builder.reset_states(self.samna_config, randomize=randomize)
                self.samna_device.get_model().apply_configuration(self.samna_config)
                # wait for the config to be written
                time.sleep(1)
                # Note: The below shouldn't be necessary ideally
                # Erase all vmem memory
                if not randomize:
                    if hasattr(self, "samna_input_graph"):
                        self.samna_input_graph.stop()
                        for lyr_idx in self.chip_layers_ordering:
                            config_builder.set_all_v_mem_to_zeros(self.samna_device, lyr_idx)
                            time.sleep(0.1)
                        self.samna_input_graph.start()
                return
        for layer in self.compatible_layers:
            if isinstance(layer, DynapcnnLayer):
                layer.spk_layer.reset_states(randomize=randomize)

    def find_chip_layer(self, layer_idx):
        """
        Given an index of a layer in the model, find the corresponding cnn core id where it is placed

        > Note that the layer index does not include the DVSLayer.
        > For instance your model comprises two layers [DVSLayer, DynapcnnLayer],
        > then the index of DynapcnnLayer is 0 and not 1.

        Parameters
        ----------
        layer_idx: int
            Index of a layer

        Returns
        -------
        chip_lyr_idx: int
            Index of the layer on the chip where the model layer is placed.
        """
        # Compute the expected number of cores
        num_cores_required = len(self.compatible_layers)
        if isinstance(self.compatible_layers[0], DVSLayer):
            num_cores_required -= 1
        if len(self.chip_layers_ordering) != num_cores_required:
            raise Exception(f"Number of layers specified in chip_layers_ordering {self.chip_layers_ordering} does not correspond to the number of cores required for this model {num_cores_required}")

        return self.chip_layers_ordering[layer_idx]

    def forward(self, x):
        if (
            hasattr(self, "device")
            and _parse_device_string(self.device)[0] in ChipFactory.supported_devices
        ):
            _ = self.samna_output_buffer.get_events()  # Flush buffer
            # NOTE: The code to start and stop time stamping is device specific
            reset_timestamps(self.device)
            enable_timestamps(self.device)
            # Send input
            self.samna_input_buffer.write(x)
            received_evts = []
            time.sleep(0.1)
            while True:
                prev_length = len(received_evts)
                time.sleep(0.1)
                received_evts.extend(self.samna_output_buffer.get_events())
                if prev_length == len(received_evts):
                    break
            # Disable timestamp
            disable_timestamps(self.device)
            return received_evts
        else:
            """Torch's forward pass."""
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


    def zero_grad(self, set_to_none: bool = False) -> None:
        for lyr in self.sequence:
            lyr.zero_grad(set_to_none)
    
    def __del__(self):
        # Stop the input graph
        if hasattr(self, 'device_input_graph') and self.device_input_graph:
            self.device_input_graph.stop()
        
        # Stop the output graph.
        if hasattr(self, 'device_output_graph') and self.device_output_graph:
            self.device_output_graph.stop()

class DynapcnnCompatibleNetwork(DynapcnnNetwork):
    """ Deprecated class, use DynapcnnNetwork instead."""

    def __init__(
        self,
        snn: Union[nn.Sequential, sinabs.Network],
        input_shape: Optional[Tuple[int, int, int]] = None,
        dvs_input: bool = False,
        discretize: bool = True,
    ):
        from warnings import warn
        warn("DynapcnnCompatibleNetwork has been renamed to DynapcnnNetwork " +
             "and will be removed in a future release.")
        super().__init__(snn, input_shape, dvs_input, discretize)
