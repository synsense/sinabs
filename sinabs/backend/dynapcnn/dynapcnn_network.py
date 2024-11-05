# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from warnings import warn

import samna
import torch
import torch.nn as nn
from samna.dynapcnn.configuration import DynapcnnConfiguration
from torch import Tensor

import sinabs
import sinabs.layers as sl

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .io import disable_timestamps, enable_timestamps, open_device, reset_timestamps
from .nir_graph_extractor import GraphExtractor
from .utils import COMPLETELY_IGNORED_LAYER_TYPES, IGNORED_LAYER_TYPES, parse_device_id
from .weight_rescaling_methods import rescale_method_1


class DynapcnnNetwork(nn.Module):
    def __init__(
        self,
        snn: nn.Module,
        input_shape: Tuple[int, int, int],
        batch_size: Optional[int] = None,
        dvs_input: Optional[bool] = None,
        discretize: bool = True,
        weight_rescaling_fn: Callable = rescale_method_1,
    ):
        """
            Given a sinabs spiking network, prepare a dynapcnn-compatible network. This can be used to
        test the network will be equivalent once on DYNAPCNN. This class also provides utilities to
        make the dynapcnn configuration and upload it to DYNAPCNN.

        Parameters
        ----------
        - snn (nn.Module): a  implementing a spiking network.
        - input_shape (tuple): a description of the input dimensions as `(features, height, width)`.
        - batch_size (optional int): If `None`, will try to infer the batch size from the model.
            If int value is provided, it has to match the actual batch size of the model.
        - dvs_input (bool): optional (default as `None`). Wether or not dynapcnn receive
            input from its DVS camera.
        - discretize (bool): If `True`, discretize the parameters and thresholds. This is needed for uploading
            weights to dynapcnn. Set to `False` only for testing purposes.
        - weight_rescaling_fn (callable): a method that handles how the re-scaling factor for one or more `SumPool2d` projecting to
            the same convolutional layer are combined/re-scaled before applying them.
        """
        super().__init__()

        # check if dvs input is expected.
        self.dvs_input = dvs_input
        self.input_shape = input_shape
        self._layer2core_map = None

        assert len(self.input_shape) == 3, "infer_input_shape did not return 3-tuple"

        # Infer batch size for dummpy input to graph extractor
        if batch_size is None:
            batch_size = sinabs.utils.get_smallest_compatible_time_dimension(snn)
        # computational graph from original PyTorch module.
        self._graph_extractor = GraphExtractor(
            snn,
            torch.randn((batch_size, *self.input_shape)),
            self.dvs_input,
            ignore_node_types=COMPLETELY_IGNORED_LAYER_TYPES,
        )

        # Remove nodes of ignored classes (including merge nodes)
        # Other than `COMPLETELY_IGNORED_LAYER_TYPES`, `IGNORED_LAYER_TYPES` are
        # part of the graph initially and are needed to ensure proper handling of
        # graph structure (e.g. Merge nodes) or meta-information (e.g.
        # `nn.Flatten` for io-shapes)
        self._graph_extractor.remove_nodes_by_class(IGNORED_LAYER_TYPES)

        # Module to execute forward pass through network
        self._dynapcnn_module = self._graph_extractor.get_dynapcnn_network_module(
            discretize=discretize, weight_rescaling_fn=weight_rescaling_fn
        )
        self._dynapcnn_module.setup_dynapcnnlayer_graph(index_layers_topologically=True)

    ####################################################### Public Methods #######################################################

    @property
    def all_layers(self):
        return self._dynapcnn_module.all_layers

    @property
    def dvs_node_info(self):
        return self._dynapcnn_module.dvs_node_info

    @property
    def chip_layers_ordering(self):
        warn(
            "`chip_layers_ordering` is deprecated. Returning `layer2core_map` instead.",
            DeprecationWarning,
        )
        return self._layer2core_map

    @property
    def dynapcnn_layers(self):
        return self._dynapcnn_module.dynapcnn_layers

    @property
    def dynapcnn_module(self):
        return self._dynapcnn_module
    
    @property
    def exit_layers(self):
        return [self.dynapcnn_layers[i] for i in self._dynapcnn_module.get_exit_layers()]

    @property
    def layer_destination_map(self):
        return self._dynapcnn_module.destination_map

    @property
    def layer2core_map(self):
        return self._layer2core_map

    @property
    def name_2_indx_map(self):
        return self._graph_extractor.name_2_indx_map

    def hw_forward(self, x):
        """Forwards data through the chip."""

        # flush buffer.
        _ = self.samna_output_buffer.get_events()

        # NOTE: The code to start and stop time stamping is device specific
        reset_timestamps(self.device)
        enable_timestamps(self.device)

        # send input.
        self.samna_input_buffer.write(x)
        received_evts = []

        # record at least until the last event has been replayed.
        min_duration = max(event.timestamp for event in x) * 1e-6
        time.sleep(min_duration)

        # keep recording if more events are being registered.
        while True:
            prev_length = len(received_evts)
            time.sleep(0.1)
            received_evts.extend(self.samna_output_buffer.get_events())
            if prev_length == len(received_evts):
                break

        # disable timestamp
        disable_timestamps(self.device)

        return received_evts

    def forward(
        self, x, return_complete: bool = False
    ) -> Union[List["event"], Tensor, Dict[int, Dict[int, Tensor]]]:
        """Forwards data through the `DynapcnnNetwork` instance.

        If the network has been deployed on a Dynapcnn/Speck device the forward
        pass happens on the devices. Otherwise the device will be simulated by
        passing the data through the `DynapcnnLayer` instances.

        Parameters
        ----------
        x: Tensor that serves as input to network. Is passed to all layers
            that are marked as entry points
        return_complete: bool that indicates whether all layer outputs should
            be return or only those with no further destinations (default)

        Returns
        -------
        The returned object depends on whether the network has been deployed
        on chip. If this is the case, a flat list of samna events is returned,
        in the order in which the events have been collected.
        If the data is passed through the `DynapcnnLayer` instances, the output
        depends on `return_complete` and on the network configuration:
        * If `return_complete` is `True`, all layer outputs will be returned in a
            dict, with layer indices as keys, and nested dicts as values, which
            hold destination indices as keys and output tensors as values.
        * If `return_complete` is `False` and there is only a single destination
            in the whole network that is marked as final (i.e. destination
            index in dynapcnn layer handler is negative), it will return the
            output as a single tensor.
        * If `return_complete` is `False` and no destination in the network
            is marked as final, a warning will be raised and the function
            returns an empty dict.
        * In all other cases a dict will be returned that is of the same
            structure as if `return_complete` is `True`, but only with entries
            where the destination is marked as final.
        """
        if (
            hasattr(self, "device")
            and parse_device_id(self.device)[0] in ChipFactory.supported_devices
        ):
            return self.hw_forward(x)
        else:
            # Forward pass through software DynapcnnLayer instance
            return self.dynapcnn_module(x, return_complete=return_complete)

    def parameters(self) -> list:
        """Gathers all the parameters of the network in a list. This is done by accessing the convolutional layer in each `DynapcnnLayer`,
        calling its `.parameters` method and saving it to a list.

        Note: the method assumes no biases are used.

        Returns
        ----------
        - parameters (list): a list of parameters of all convolutional layers in the `DynapcnnNetwok`.
        """
        parameters = []

        for layer in self.dynapcnn_layers.values():
            if isinstance(layer, DynapcnnLayer):
                parameters.extend(layer.conv_layer.parameters())

        return parameters

    def memory_summary(self) -> Dict[str, Dict[int, int]]:
        """Get a summary of the network's memory requirements.

        Returns
        -------
        dict:
            A dictionary with keys kernel, neuron, bias. The values are a dicts.
            Each nested dict has as keys the indices of all dynapcnn_layers and
            as values the corresonding memory values for each layer.
        """
        # For each entry (kernel, neuron, bias) provide one nested dict with 
        # one entry for each layer
        summary = {key: dict() for key in ("kernel", "neuron", "bias")}

        for layer_index, layer in self.dynapcnn_layers.items():
            for key, val in layer.memory_summary().items():
                summary[key][layer_index] = val

        return summary

    def init_weights(self, init_fn: nn.init = nn.init.xavier_normal_) -> None:
        """Call the weight initialization method `init_fn` on each `DynapcnnLayer.conv_layer.weight.data` in the `DynapcnnNetwork` instance.

        Parameters
        ----------
        - init_fn (torch.nn.init): the weight initialization method to be used.
        """
        for layer in self.dynapcnn_layers.values():
            if isinstance(layer, DynapcnnLayer):
                init_fn(layer.conv_layer.weight.data)

    def detach_neuron_states(self) -> None:
        """Detach the neuron states and activations from current computation graph (necessary)."""

        for module in self.dynapcnn_layers.values():
            if isinstance(module, DynapcnnLayer):
                if isinstance(module.spk_layer, sl.StatefulLayer):
                    for name, buffer in module.spk_layer.named_buffers():
                        buffer.detach_()

    def to(
        self,
        device: str = "cpu",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier: Optional[Callable] = None,
        slow_clk_frequency: Optional[int] = None,
        layer2core_map: Union[Dict[int, int], str] = "auto",
        chip_layers_ordering: Optional[Union[Sequence[int], str]] = None,
    ):
        """Deploy model to cpu, gpu or a SynSense device.

        Note that the model parameters are only ever transferred to the device on the `to` call,
        so changing a threshold or weight of a model that is deployed will have no effect on the
        model on chip until `to` is called again.

        Parameters
        ----------

        device: String
            cpu:0, cuda:0, dynapcnndevkit, speck2devkit

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

        layer2core_map (dict or "auto"): Defines how cores on chip are
            assigned to DynapcnnLayers. If `auto`, an automated procedure
            will be used to find a valid ordering. Otherwise a dict needs
            to be passed, with DynapcnnLayer indices as keys and assigned
            core IDs as values. DynapcnnLayer indices have to match those of
            `self.dynapcnn_layers`.

        chip_layers_ordering: sequence of integers or `auto`
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            The index of the core on chip to which the i-th layer in the model is mapped is the value of the i-th entry in the list.
            Note: This list should be the same length as the number of dynapcnn layers in your model.
            Note: This parameter is obsolete and should not be passed anymore. Use
            `layer2core_map` instead.

        Note
        ----
        chip_layers_ordering and monitor_layers are used only when using synsense devices.
        For GPU or CPU usage these options are ignored.
        """
        self.device = device

        if isinstance(device, torch.device):
            self._to_device(device)

        elif isinstance(device, str):
            device_name, _ = parse_device_id(device)

            if device_name in ChipFactory.supported_devices:

                # generate config.
                config = self.make_config(
                    layer2core_map=layer2core_map,
                    chip_layers_ordering=chip_layers_ordering,
                    device=device,
                    monitor_layers=monitor_layers,
                    config_modifier=config_modifier,
                )

                # apply configuration to device.
                self.samna_device = open_device(device)
                self.samna_device.get_model().apply_configuration(config)
                time.sleep(1)

                # set external slow-clock if needed.
                if slow_clk_frequency is not None:
                    dk_io = self.samna_device.get_io_module()
                    dk_io.set_slow_clk(True)
                    dk_io.set_slow_clk_rate(slow_clk_frequency)  # Hz

                builder = ChipFactory(device).get_config_builder()

                # create input source node.
                self.samna_input_buffer = builder.get_input_buffer()

                # create output sink node node.
                self.samna_output_buffer = builder.get_output_buffer()

                # connect source node to device sink.
                self.device_input_graph = samna.graph.EventFilterGraph()
                self.device_input_graph.sequential(
                    [
                        self.samna_input_buffer,
                        self.samna_device.get_model().get_sink_node(),
                    ]
                )

                # connect sink node to device.
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
                self._to_device(device)

        else:
            raise Exception("Unknown device description.")

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
            if e.args[0] == (
                "One or more of the DynapcnnLayers could not be mapped to any core."
            ):
                return False
            else:
                raise e
        return is_compatible

    def make_config(
        self,
        layer2core_map: Union[Dict[int, int], str] = "auto",
        device: str = "dynapcnndevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier: Optional[Callable] = None,
        chip_layers_ordering: Optional[Union[Sequence[int], str]] = None,
    ) -> DynapcnnConfiguration:
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

        Parameters
        ----------
        - layer2core_map (dict or "auto"): Defines how cores on chip are
            assigned to DynapcnnLayers. If `auto`, an automated procedure
            will be used to find a valid ordering. Otherwise a dict needs
            to be passed, with DynapcnnLayer indices as keys and assigned
            core IDs as values. DynapcnnLayer indices have to match those of
            `self.dynapcnn_layers`.
        - device: (string): dynapcnndevkit, speck2b or speck2devkit
        - monitor_layers: None/List/Str
            A list of all layers in the module that you want to monitor. Indexing starts with the first non-dvs layer.
            If you want to monitor the dvs layer for eg.
            ::

                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                monitor_layers = "all" # If you want to monitor all the layers

            If this value is left as None, by default the last layer of the model is monitored.

        - config_modifier (Callable or None):
            A user configuration modifier method.
            This function can be used to make any custom changes you want to make to the configuration object.
        - chip_layers_ordering (None, sequence of integers or "auto", obsolete):
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            Note: This list should be the same length as the number of dynapcnn layers in your model.
            Note: This parameter is obsolete and should not be passed anymore. Use
            `layer2core_map` instead.

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
            layer2core_map=layer2core_map,
            device=device,
            monitor_layers=monitor_layers,
            config_modifier=config_modifier,
            chip_layers_ordering=chip_layers_ordering,
        )

        # Validate config
        if is_compatible:
            print("Network is valid")
            return config
        else:
            raise ValueError(f"Generated config is not valid for {device}")

    def has_dvs_layer(self) -> bool:
        """Return True if there is a DVSLayer in the network

        Returns
        -------
        bool: True if DVSLayer is found within the network.
        """
        for layer in self.dynapcnn_layers.values():
            if isinstance(layer, DVSLayer):
                return True
        return False

    ####################################################### Private Methods #######################################################

    def _make_config(
        self,
        layer2core_map: Union[Dict[int, int], str] = "auto",
        device: str = "dynapcnndevkit:0",
        monitor_layers: Optional[Union[List, str]] = None,
        config_modifier: Optional[Callable] = None,
        chip_layers_ordering: Optional[Union[Sequence[int], str]] = None,
    ) -> Tuple[DynapcnnConfiguration, bool]:
        """Prepare and output the `samna` DYNAPCNN configuration for this network.

        Parameters
        ----------
        - layer2core_map (dict or "auto"): Defines how cores on chip are
            assigned to DynapcnnLayers. If `auto`, an automated procedure
            will be used to find a valid ordering. Otherwise a dict needs
            to be passed, with DynapcnnLayer indices as keys and assigned
            core IDs as values. DynapcnnLayer indices have to match those of
            `self.dynapcnn_layers`.
        - device: (string): dynapcnndevkit, speck2b or speck2devkit
        - monitor_layers: None/List/Str
            A list of all layers in the module that you want to monitor. Indexing starts with the first non-dvs layer.
            If you want to monitor the dvs layer for eg.
            ::

                monitor_layers = ["dvs"]  # If you want to monitor the output of the pre-processing layer
                monitor_layers = ["dvs", 8] # If you want to monitor preprocessing and layer 8
                monitor_layers = "all" # If you want to monitor all the layers

            If this value is left as None, by default the last layer of the model is monitored.

        - config_modifier (Callable or None):
            A user configuration modifier method.
            This function can be used to make any custom changes you want to make to the configuration object.
        - chip_layers_ordering (None, sequence of integers or "auto", obsolete):
            The order in which the dynapcnn layers will be used. If `auto`,
            an automated procedure will be used to find a valid ordering.
            A list of layers on the device where you want each of the model's DynapcnnLayers to be placed.
            Note: This list should be the same length as the number of dynapcnn layers in your model.
            Note: This parameter is obsolete and should not be passed anymore. Use
            `layer2core_map` instead.

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
            ValueError
                If no valid mapping between the layers of this object and the cores of
                the provided device can be found.
        """
        config_builder = ChipFactory(device).get_config_builder()

        if chip_layers_ordering is not None:
            if layer2core_map is not None:
                warn(
                    "Both `chip_layers_ordering` and `layer2core_map are provided. "
                    "Please only provide `layer2core_map`, as `chip_layers_ordering` "
                    "is deprecated.",
                    DeprecationWarning,
                )
            elif chip_layers_ordering == "auto":
                warn(
                    "The parameter `chip_layers_ordering` is deprecated. Passing "
                    "'auto' is still accepted, but in the future please use "
                    "`layer2core_map` instead.",
                    DeprecationWarning,
                )
                layer2core_map = "auto"
            else:
                raise ValueError(
                    "`chip_layers_ordering` is deprecated. Passing anything other "
                    "than `None` or 'auto' is not possible. To manually assign core "
                    "to layers, please use the `layer2core_map` argument."
                )
        if layer2core_map == "auto":
            # Assign chip core ID for each DynapcnnLayer.
            layer2core_map = config_builder.map_layers_to_cores(self.dynapcnn_layers)
        else:
            if not layer2core_map.keys() == self.dynapcnn_layers.keys():
                raise ValueError(
                    "The keys provided in `layer2core_map` must exactly match "
                    "the keys in `self.dynapcnn_layers`"
                )

        self._layer2core_map = layer2core_map

        # update config (config. DynapcnnLayer instances into their assigned core).
        config = config_builder.build_config(
            layers=self.all_layers,
            destination_map=self.layer_destination_map,
            layer2core_map=layer2core_map,
        )

        if monitor_layers is None:
            # Monitor all layers with exit point destinations
            monitor_layers = self._dynapcnn_module.get_exit_layers()

        elif monitor_layers == "all":
            monitor_layers = [
                lyr_idx
                for lyr_idx, layer in self.dynapcnn_layers.items()
                if not isinstance(layer, DVSLayer)
            ]

        # Collect cores (chip layers) that are to be monitored
        monitor_chip_layers = []
        for lyr_idx in monitor_layers:
            if str(lyr_idx).lower() == "dvs":
                monitor_chip_layers.append("dvs")
            else:
                monitor_chip_layers.append(layer2core_map[lyr_idx])

        # enable monitors on the specified layers.
        config_builder.monitor_layers(config, monitor_chip_layers)

        if config_modifier is not None:
            # apply user config modifier.
            config = config_modifier(config)

        # Validate config
        return config, config_builder.validate_configuration(config)

    def _to_device(self, device: torch.device) -> None:
        """Access each sub-layer within all `DynapcnnLayer` instances and call `.to(device)` on them."""
        for layer in self.dynapcnn_layers.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer.DynapcnnLayer):
                layer.to(device)

        for _, data in self._merge_points.items():
            data["merge"].to(device)

    def __str__(self):
        pretty_print = ""
        for idx, layer_data in self.dynapcnn_layers.items():
            pretty_print += f"----------------------- [ DynapcnnLayer {idx} ] -----------------------\n"
            pretty_print += f"{layer_data}\n\n"

        return pretty_print


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
