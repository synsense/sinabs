# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import samna
import torch
import torch.nn as nn
from torch import Tensor

import sinabs
import sinabs.layers as sl

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .dynapcnn_layer_utils import construct_dynapcnnlayers_from_mapper
from .dynapcnnnetwork_module import DynapcnnNetworkModule
from .io import disable_timestamps, enable_timestamps, open_device, reset_timestamps
from .nir_graph_extractor import GraphExtractor
from .sinabs_edges_handler import collect_dynapcnn_layer_info
from .utils import (
    DEFAULT_IGNORED_LAYER_TYPES,
    parse_device_id,
)
from .weight_rescaling_methods import rescale_method_1


class DynapcnnNetwork(nn.Module):
    def __init__(
        self,
        snn: nn.Module,
        input_shape: Tuple[int, int, int],
        batch_size: int,
        dvs_input: bool = False,
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
        - dvs_input (bool): wether or not dynapcnn receive input from its DVS camera.
        - discretize (bool): If `True`, discretize the parameters and thresholds. This is needed for uploading
            weights to dynapcnn. Set to `False` only for testing purposes.
        - weight_rescaling_fn (callable): a method that handles how the re-scaling factor for one or more `SumPool2d` projecting to
            the same convolutional layer are combined/re-scaled before applying them.

        Notes
        ----------
        Some of the properties defined within the class constructor are meant to be temporary data structures handling the conversion
        of the `snn` (the original `nn.Module`) into a set of `DynapcnnLayer`s composing a `DynapcnnNetwork` instance. Once their role
        in preprocessing `snn` is finished, all required data to train/deploy the `DynapcnnNetwork` instance is within `self._dcnnl_edges`
        (the connectivity between each `DynapcnnLayer`/core), `self._layers_mapper` (every `DynapcnnLayer` in the network) and `self._merge_points`
        (the `DynapcnnLayer`s that need a `Merge` input). Thus, the following private properties are delted as last step of the constructor:

        - self._graph_extractor
        - self._sinabs_edges
        - self._sinabs_indx_2_module_map
        - self._dcnnl_map
        - self._dynapcnn_layers
        """
        super().__init__()

        # TODO for now the graph part is not taking into consideration DVS inputs.
        # check if dvs input is expected.
        dvs_input = False
        self.dvs_input = dvs_input
        self.input_shape = input_shape

        assert len(self.input_shape) == 3, "infer_input_shape did not return 3-tuple"

        # computational graph from original PyTorch module.
        self._graph_extractor = GraphExtractor(
            snn, torch.randn((batch_size, *self.input_shape))
        )  # needs the batch dimension.

        # Remove nodes of ignored classes (including merge nodes)
        self._graph_extractor.remove_nodes_by_class(DEFAULT_IGNORED_LAYER_TYPES)

        # create a dict holding the data necessary to instantiate a `DynapcnnLayer`.
        self._dcnnl_map = collect_dynapcnn_layer_info(
            self._graph_extractor.indx_2_module_map,
            self._graph_extractor.edges,
            self._graph_extractor.nodes_io_shapes,
            self._graph_extractor.entry_nodes,
        )

        # build `DynapcnnLayer` instances from mapper.
        dynapcnn_layers, destination_map, entry_points = (
            construct_dynapcnnlayers_from_mapper(
                dcnnl_map=self._dcnnl_map,
                discretize=discretize,
                rescale_fn=weight_rescaling_fn,
            )
        )

        # Module to execute forward pass through network
        self._dynapcnn_module = DynapcnnNetworkModule(
            dynapcnn_layers, destination_map, entry_points
        )
        self.dynapcnn_module.setup_dynapcnnlayer_graph(index_layers_topologically=True)

    ####################################################### Public Methods #######################################################

    @property
    def dynapcnn_layers(self):
        return self._dynapcnn_module.dynapcnn_layers

    @property
    def dynapcnn_module(self):
        return self._dynapcnn_module

    @property
    def chip_layers_ordering(self):
        return self._chip_layers_ordering

    def get_output_core_id(self) -> int:
        """."""

        # TODO if a network with two output layers is deployed, which is not supported yet btw, this monitoring part needs to be revised.
        for _, ith_dcnnl in self._layers_mapper.items():
            if len(ith_dcnnl.dynapcnnlayer_destination) == 0:
                # a DynapcnnLayer without destinations is taken to be the output layer of the network.
                return ith_dcnnl.assigned_core

    def get_input_core_id(self) -> list:
        """Since the chip allows for multiple input layers (that merge into a single output at some point), this method returns
        a list of all core IDs to which an input layer of the network has been assigned to.
        """
        entry_points = []
        for _, ith_dcnnl in self._layers_mapper.items():
            if ith_dcnnl.entry_point:
                entry_points.append(ith_dcnnl.assigned_core)

        return entry_points

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

        for layer in self._layers_mapper.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer.DynapcnnLayer):
                parameters.extend(layer.conv_layer.parameters())

        return parameters

    def init_weights(self, init_fn: nn.init = nn.init.xavier_normal_) -> None:
        """Call the weight initialization method `init_fn` on each `DynapcnnLayer.conv_layer.weight.data` in the `DynapcnnNetwork` instance.

        Parameters
        ----------
        - init_fn (torch.nn.init): the weight initialization method to be used.
        """
        for layer in self._layers_mapper.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer.DynapcnnLayer):
                init_fn(layer.conv_layer.weight.data)

    def detach_neuron_states(self) -> None:
        """Detach the neuron states and activations from current computation graph (necessary)."""

        for module in self._layers_mapper.values():
            if isinstance(module, sinabs.backend.dynapcnn.dynapcnn_layer.DynapcnnLayer):
                if isinstance(module.spk_layer, sl.StatefulLayer):
                    for name, buffer in module.spk_layer.named_buffers():
                        buffer.detach_()

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
            self._to_device(device)

        elif isinstance(device, str):
            device_name, _ = parse_device_id(device)

            if device_name in ChipFactory.supported_devices:

                # generate config.
                config = self._make_config(
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

    ####################################################### Private Methods #######################################################

    def _make_config(
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
        config_builder = ChipFactory(device).get_config_builder()

        # TODO not handling DVSLayer yet.
        has_dvs_layer = isinstance(self._layers_mapper[0], DVSLayer)

        if chip_layers_ordering == "auto":
            # figure out mapping of each `DynapcnnLayer` into one core (core ID will be set in the layer's handler instance via `.assigned_core`).
            _ = config_builder.get_valid_mapping(self)

        else:
            # TODO - mapping from each DynapcnnLayer into cores has been provided by the user: NOT IMPLEMENTED YET.
            if has_dvs_layer:
                # TODO not handling DVSLayer yet.
                pass

        # update config (config. DynapcnnLayer instances into their assigned core).
        config = config_builder.build_config(self)

        # TODO not handling DVSLayer yet (this is from the old implementation, should be revised).
        if self.input_shape and self.input_shape[0] == 1:
            config.dvs_layer.merge = True

        # TODO all this monitoring part needs validation still.
        monitor_chip_layers = []
        if monitor_layers is None:
            # check if any monitoring is enabled (if not, enable monitoring for the last layer).
            for dcnnl_index, ith_dcnnl in self._layers_mapper.items():

                # TODO if a network with two output layers is deployed, which is not supported yet btw, this monitoring part needs to be revised.
                if (
                    len(
                        self._dynapcnnlayers_handlers[
                            dcnnl_index
                        ].dynapcnnlayer_destination
                    )
                    == 0
                ):
                    # a DynapcnnLayer without destinations is taken to be the output layer of the network.
                    monitor_chip_layers.append(
                        self._dynapcnnlayers_handlers[dcnnl_index].assigned_core
                    )

        elif monitor_layers == "all":
            for dcnnl_index, ith_dcnnl in self._layers_mapper.items():
                # TODO not handling DVSLayer yet
                # monitor each chip core (if not a DVSLayer).
                if not isinstance(ith_dcnnl, DVSLayer):
                    monitor_chip_layers.append(
                        self._dynapcnnlayers_handlers[dcnnl_index].assigned_core
                    )

        if monitor_layers:
            if "dvs" in monitor_layers:
                monitor_chip_layers.append("dvs")

        # enable monitors on the specified layers.
        config_builder.monitor_layers(config, monitor_chip_layers)

        if config_modifier is not None:
            # apply user config modifier.
            config = config_modifier(config)

        if config_builder.validate_configuration(config):
            # validate config.
            print("Network is valid: \n")

            return config
        else:
            raise ValueError(f"Generated config is not valid for {device}")

    def _to_device(self, device: torch.device) -> None:
        """Access each sub-layer within all `DynapcnnLayer` instances and call `.to(device)` on them."""
        for layer in self._layers_mapper.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer.DynapcnnLayer):
                layer.to(device)

        for _, data in self._merge_points.items():
            data["merge"].to(device)

    def __str__(self):
        pretty_print = ""
        for idx, layer_data in self._layers_mapper.items():
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
