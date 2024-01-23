import time
from subprocess import CalledProcessError
from typing import List, Optional, Sequence, Tuple, Union

import samna
import torch
import torch.nn as nn

import sinabs

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .io import disable_timestamps, enable_timestamps, open_device, reset_timestamps
from .utils import (
    DEFAULT_IGNORED_LAYER_TYPES,
    build_from_list,
    convert_model_to_layer_list,
    infer_input_shape,
    parse_device_id,
)


class DynapcnnNetwork(nn.Module):
    """Given a sinabs spiking network, prepare a dynapcnn-compatible network. This can be used to
    test the network will be equivalent once on DYNAPCNN. This class also provides utilities to
    make the dynapcnn configuration and upload it to DYNAPCNN.

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

    Note that the model parameters are only ever transferred to the device
    on the `to` call, so changing a threshold or weight of a model that
    is deployed will have no effect on the model on chip until `to` is called again.
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

        self.input_shape = input_shape  # Convert models  to sequential
        layers = convert_model_to_layer_list(
            model=snn, ignore=DEFAULT_IGNORED_LAYER_TYPES
        )
        # Check if dvs input is expected
        if dvs_input:
            self.dvs_input = True
        else:
            self.dvs_input = False

        input_shape = infer_input_shape(layers, input_shape=input_shape)
        assert len(input_shape) == 3, "infer_input_shape did not return 3-tuple"

        # Build model from layers
        self.sequence = build_from_list(
            layers,
            in_shape=input_shape,
            discretize=discretize,
            dvs_input=self.dvs_input,
        )

    def to(
        self,
        device="cpu",
        chip_layers_ordering="auto",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
        slow_clk_frequency: int = None,
    ):
        """Note that the model parameters are only ever transferred to the device on the `to` call,
        so changing a threshold or weight of a model that is deployed will have no effect on the
        model on chip until `to` is called again.

        Parameters
        ----------

        device: String
            cpu:0, cuda:0, dynapcnndevkit, speck2devkit

        chip_layers_ordering: sequence of integers or `auto`
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            The index of the core on chip to which the i-th layer in the model is mapped is the value of the i-th entry in the list.
            Note: This list should be the same length as the number of dynapcnn layers in your model.

        monitor_layers: None/List
            A list of all layers in the module that you want to monitor. Indexing starts with the first non-dvs layer.
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
            device_name, _ = parse_device_id(device)
            if device_name in ChipFactory.supported_devices:  # pragma: no cover
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

                # Set external slow-clock if need
                if slow_clk_frequency is not None:
                    dk_io = self.samna_device.get_io_module()
                    dk_io.set_slow_clk(True)
                    dk_io.set_slow_clk_rate(slow_clk_frequency)  # Hz

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
                        self.samna_device.get_model().get_sink_node(),
                    ]
                )

                # Connect sink node to device
                self.device_output_graph = samna.graph.EventFilterGraph()
                self.device_output_graph.sequential(
                    [
                        self.samna_device.get_model().get_source_node(),
                        self.samna_output_buffer,
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

    def _make_config(
        self,
        chip_layers_ordering: Union[Sequence[int], str] = "auto",
        device="dynapcnndevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
    ) -> Tuple["SamnaConfiguration", bool]:
        """Prepare and output the `samna` configuration for this network.

        Parameters
        ----------

        chip_layers_ordering: sequence of integers or `auto`
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            The index of the core on chip to which the i-th layer in the model is mapped is the value of the i-th entry in the list.
            Note: This list should be the same length as the number of dynapcnn layers in your model.

        device: String
            dynapcnndevkit, speck2b or speck2devkit

        monitor_layers: None/List/Str
            A list of all layers in the module that you want to monitor. Indexing starts with the first non-dvs layer.
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
        Bool
            True if the configuration is valid for the given device.

        Raises
        ------
            ImportError
                If samna is not available.
        """
        config_builder = ChipFactory(device).get_config_builder()

        has_dvs_layer = isinstance(self.sequence[0], DVSLayer)

        # Figure out layer ordering
        if chip_layers_ordering == "auto":
            chip_layers_ordering = config_builder.get_valid_mapping(self)
        else:
            # Truncate chip_layers_ordering just in case a longer list is passed
            if has_dvs_layer:
                chip_layers_ordering = chip_layers_ordering[: len(self.sequence) - 1]
            chip_layers_ordering = chip_layers_ordering[: len(self.sequence)]

        # Save the chip layers
        self.chip_layers_ordering = chip_layers_ordering
        # Update config
        config = config_builder.build_config(self, chip_layers_ordering)
        if self.input_shape and self.input_shape[0] == 1:
            config.dvs_layer.merge = True
        # Check if any monitoring is enabled and if not, enable monitoring for the last layer
        if monitor_layers is None:
            monitor_layers = [-1]
        elif monitor_layers == "all":
            num_cnn_layers = len(self.sequence) - int(has_dvs_layer)
            monitor_layers = list(range(num_cnn_layers))

        # Enable monitors on the specified layers
        # Find layers corresponding to the chip
        monitor_chip_layers = [
            self.find_chip_layer(lyr) for lyr in monitor_layers if lyr != "dvs"
        ]
        if "dvs" in monitor_layers:
            monitor_chip_layers.append("dvs")
        config_builder.monitor_layers(config, monitor_chip_layers)

        # Fix default factory setting to not return input events (UGLY!! Ideally this should happen in samna)
        # config.factory_settings.monitor_input_enable = False

        # Apply user config modifier
        if config_modifier is not None:
            config = config_modifier(config)

        # Validate config
        return config, config_builder.validate_configuration(config)

    def make_config(
        self,
        chip_layers_ordering: Union[Sequence[int], str] = "auto",
        device="dynapcnndevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier=None,
    ):
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

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
            A list of all layers in the module that you want to monitor. Indexing starts with the first non-dvs layer.
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
            ValueError
                If the generated configuration is not valid for the specified device.
        """
        config, is_compatible = self._make_config(
            chip_layers_ordering=chip_layers_ordering,
            device=device,
            monitor_layers=monitor_layers,
            config_modifier=config_modifier,
        )
        # Validate config
        if is_compatible:
            print("Network is valid")
            return config
        else:
            raise ValueError(f"Generated config is not valid for {device}")

    def is_compatible_with(self, device_type: str) -> bool:
        """Check if the current model is compatible with a given device.

        Args:
            device_type (str): Device type ie speck2b, speck2fmodule

        Returns:
            bool: True if compatible
        """
        try:
            _, is_compatible = self._make_config(device=device_type)
        except ValueError as e:
            # Catch "No valid mapping found" error
            if e.args[0] == ("No valid mapping found"):
                return False
            else:
                raise e
        return is_compatible

    def reset_states(self, randomize=False):
        """Reset the states of the network."""
        if hasattr(self, "device") and isinstance(self.device, str):  # pragma: no cover
            device_name, _ = parse_device_id(self.device)
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
                            config_builder.set_all_v_mem_to_zeros(
                                self.samna_device, lyr_idx
                            )
                            time.sleep(0.1)
                        self.samna_input_graph.start()
                return
        for layer in self.sequence:
            if isinstance(layer, DynapcnnLayer):
                layer.spk_layer.reset_states(randomize=randomize)

    def find_chip_layer(self, layer_idx):
        """Given an index of a layer in the model, find the corresponding cnn core id where it is
        placed.

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
        num_cores_required = len(self.sequence)
        if isinstance(self.sequence[0], DVSLayer):
            num_cores_required -= 1
        if len(self.chip_layers_ordering) != num_cores_required:
            raise Exception(
                f"Number of layers specified in chip_layers_ordering {self.chip_layers_ordering} does not correspond to the number of cores required for this model {num_cores_required}"
            )

        return self.chip_layers_ordering[layer_idx]

    def forward(self, x):
        if (
            hasattr(self, "device")
            and parse_device_id(self.device)[0] in ChipFactory.supported_devices
        ):  # pragma: no cover
            _ = self.samna_output_buffer.get_events()  # Flush buffer
            # NOTE: The code to start and stop time stamping is device specific
            reset_timestamps(self.device)
            enable_timestamps(self.device)
            # Send input
            self.samna_input_buffer.write(x)
            received_evts = []
            # Record at least until the last event has been replayed
            min_duration = max(event.timestamp for event in x) * 1e-6
            time.sleep(min_duration)
            # Keep recording if more events are being registered
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
        """Get a summary of the network's memory requirements.

        Returns
        -------
        dict:
            A dictionary with keys kernel, neuron, bias.
            The values are a list of the corresponding number per layer in the same order as the model
        """
        summary = {}

        dynapcnn_layers = [
            lyr for lyr in self.sequence if isinstance(lyr, DynapcnnLayer)
        ]
        summary.update({k: list() for k in dynapcnn_layers[0].memory_summary().keys()})
        for lyr in dynapcnn_layers:
            lyr_summary = lyr.memory_summary()
            for k, v in lyr_summary.items():
                summary[k].append(v)
        return summary

    def zero_grad(self, set_to_none: bool = False) -> None:
        for lyr in self.sequence:
            lyr.zero_grad(set_to_none)

    def __del__(self):
        # Stop the input graph
        if hasattr(self, "device_input_graph") and self.device_input_graph:
            self.device_input_graph.stop()

        # Stop the output graph.
        if hasattr(self, "device_output_graph") and self.device_output_graph:
            self.device_output_graph.stop()


class DynapcnnCompatibleNetwork(DynapcnnNetwork):
    """Deprecated class, use DynapcnnNetwork instead."""

    def __init__(
        self,
        snn: Union[nn.Sequential, sinabs.Network],
        input_shape: Optional[Tuple[int, int, int]] = None,
        dvs_input: bool = False,
        discretize: bool = True,
    ):
        from warnings import warn

        warn(
            "DynapcnnCompatibleNetwork has been renamed to DynapcnnNetwork "
            + "and will be removed in a future release."
        )
        super().__init__(snn, input_shape, dvs_input, discretize)
