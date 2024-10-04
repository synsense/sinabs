# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import samna
import torch
import torch.nn as nn

import sinabs
import sinabs.layers as sl

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .dynapcnnnetwork_module import DynapcnnNetworkModule
from .io import disable_timestamps, enable_timestamps, open_device, reset_timestamps
from .nir_graph_extractor import GraphExtractor
from .sinabs_edges_handler import collect_dynapcnn_layer_info
from .utils import (
    DEFAULT_IGNORED_LAYER_TYPES,
    Edge,
    build_from_graph,
    parse_device_id,
    topological_sorting,
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
        - self._nodes_to_dcnnl_map
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
        self._nodes_to_dcnnl_map = collect_dynapcnn_layer_info(
            self._graph_extractor.indx_2_module_map,
            self._graph_extractor.edges,
            self._graph_extractor.nodes_io_shapes,
        )

        # TODO: Try avoiding this step by only retrieving input shapes of conv layers
        # while constractung dcnnl_map
        # # updates 'self._nodes_to_dcnnl_map' to include the I/O shape for each node.
        # self._populate_nodes_io()

        # build `DynapcnnLayer` instances from graph edges and mapper.
        self._dynapcnn_layers, self._dynapcnnlayers_handlers = build_from_graph(
            discretize=discretize,
            edges=self._graph_extractor.edges,
            nodes_to_dcnnl_map=self._nodes_to_dcnnl_map,
            weight_rescaling_fn=weight_rescaling_fn,
            entry_nodes=self._graph_extractor._entry_nodes,
        )

        # these gather all data necessay to implement the forward method for this class.
        (
            self._dcnnl_edges,
            self._layers_mapper,
            self._merge_points,
            self._topological_order,
        ) = self._get_network_module()

        # all necessary `DynapcnnLayer` data held in `self._layers_mapper`: removing intermediary data structures no longer necessary.
        del self._graph_extractor
        del self._sinabs_edges
        del self._sinabs_indx_2_module_map
        del self._nodes_to_dcnnl_map
        del self._dynapcnn_layers

    ####################################################### Public Methods #######################################################

    @property
    def dcnnl_edges(self):
        return self._dcnnl_edges

    @property
    def merge_points(self):
        return self._merge_points

    @property
    def topological_order(self):
        return self._topological_order

    @property
    def layers_mapper(self) -> Dict[int, DynapcnnLayer]:
        return self._layers_mapper

    @property
    def layers_handlers(self):
        return self._dynapcnnlayers_handlers

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

    def forward(self, x):
        """Forwards data through the `DynapcnnNetwork` instance. This method relies on three main data structures created to represent
        the `DynapcnnLayer`s in the network and the data propagation through them during the forward pass:

        - `self._topological_order` (list): this is used to guide the sequence in which the `DynapcnnLayer`s in `self._layers_mapper` are to be called
            to generate the input tensors to be propagated through the network during the forward pass.
        - `self._dcnnl_edges` (list): this list of edges represent the graph describing the interactions between each `DynapcnnLayer` (the nodes in
            the edges are the indices of these layers). An `edge` is used to index a mapper (using `edge[0]`) in order to retrieve the output to be fed
            as input to a `DynapcnnLayer` instance (indexed by `edge[1]`).
        - `self._layers_mapper` (dict): a mapper used to forward data through the `DynapcnnNetwork` instances. Each `key` is the indice associated
            with a `DynapcnnLayer` instance.
        - `self._merge_points` (dict): this mapper has a "support" role. It indexes wich convolutional layers in the set of `DynapcnnLayer`s
            composing the network require two sources of input (because their input tensor is the output of a `Merge` layer).
        """

        layers_outputs = {}

        for i in self._topological_order:

            if self._dynapcnnlayers_handlers[i].entry_point:
                # `DynapcnnLayer i` is an entry point of the network.
                layers_outputs[i] = self._layers_mapper[i](x)

            else:
                # input to `DynapcnnLayer i` is the output of another instance.

                if i in self._merge_points and i not in layers_outputs:
                    # there are two sources of input for `DynapcnnLayer i`.

                    # by this points the arguments of the `Merge` associated with `i` should have been computed due to the topological sorting.
                    arg1, arg2 = self._merge_points[i]["sources"]

                    #   find which returned tensor from the `forward` call of DynapcnnLayers `arg1` and `arg2` are to be fed
                    # to the target DynapcnnLayer `i`.
                    return_index_arg1 = self._dynapcnnlayers_handlers[
                        arg1
                    ].get_destination_dcnnl_index(i)
                    return_index_arg2 = self._dynapcnnlayers_handlers[
                        arg2
                    ].get_destination_dcnnl_index(i)

                    # retrieve input tensors to `Merge`.
                    _arg1 = layers_outputs[arg1][return_index_arg1]
                    _arg2 = layers_outputs[arg2][return_index_arg2]

                    # merge tensors.
                    merge_output = self._merge_points[i]["merge"](_arg1, _arg2)

                    # call the forward.
                    layers_outputs[i] = self._layers_mapper[i](merge_output)

                else:
                    # there's a single source of input for `DynapcnnLayer i`.

                    # input source for `i`.
                    src_dcnnl = self._get_input_to_dcnnl(i)

                    #   find which returned tensor from the `forward` call of the source DynapcnnLayer `src_dcnnl` is to be fed
                    # to the target DynapcnnLayer `i`.
                    return_index = self._dynapcnnlayers_handlers[
                        src_dcnnl
                    ].get_destination_dcnnl_index(i)

                    # call the forward.
                    layers_outputs[i] = self._layers_mapper[i](
                        layers_outputs[src_dcnnl][return_index]
                    )

        # TODO - this assumes the network has a single output node.
        return layers_outputs[self._topological_order[-1]][0]

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

    def _get_input_to_dcnnl(self, dcnnl_ID) -> int:
        """Returns the ID of the first `DynapcnnLayer` forwarding its input to `dcnnl_ID`."""
        for edge in self._dcnnl_edges:
            if edge[1] == dcnnl_ID:
                return edge[0]
        raise ValueError(f"DynapcnnLayer {dcnnl_ID} has no source of input.")

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

    def _get_network_module(self) -> Union[list, dict, dict]:
        """Uses the `DynapcnnLayer` instances in `self._dynapcnn_layers` and the connectivity between them to create three data structures
        that guide the data forwarding between the layer during the forward pass.

        Returns
        ----------
        - dcnnl_edges (list): edges, represented as tuples of `DynapcnnLayer` indices, used to guide the data forwarding through each `DynapcnnLayer` in forward method.
        - forward_map (dict): have all the `DynapcnnLayer` (`value`), each being accessible via its index (`key`). Used to call `DynapcnnLayer.forward` in forward method.
        - merge_points (dict): used to compose the inputs to a `DynapcnnLayer` that requires an input from a `Merge` layer.

        Notes
        ----------
        - the property `DynapcnnLayer.assigned_core` is only set after `self.to(device='speck...')` is called.
        """

        # get connections between `DynapcnnLayer`s.
        dcnnl_edges = self._get_dynapcnnlayers_edges()

        dcnnnet_module = DynapcnnNetworkModule(
            dcnnl_edges, self._dynapcnn_layers, self._dynapcnnlayers_handlers
        )

        return (
            dcnnnet_module.dcnnl_edges,
            dcnnnet_module.forward_map,
            dcnnnet_module.merge_points,
            topological_sorting(dcnnl_edges),
        )

    def _get_dynapcnnlayers_edges(self) -> List[Edge]:
        """Create edges representing connections between `DynapcnnLayer` instances.

        Returns
        ----------
        - dcnnl_edges (list): a list of edges using the IDs of `DynapcnnLayer` instances. These edges describe the computational
            graph implemented by the layers of the model (i.e., how the `DynapcnnLayer` instances address each other).
        """
        dcnnl_edges = []

        for dcnnl_idx, layer_data in self._dynapcnn_layers.items():
            for dest in layer_data["destinations"]:
                dcnnl_edges.append((dcnnl_idx, dest))

        return dcnnl_edges

    def _populate_nodes_io(self):
        """Loops through the nodes in the original graph to retrieve their I/O tensor shapes and add them to their respective
        representations in `self._nodes_to_dcnnl_map`."""

        def find_my_input(edges_list: list, node: int) -> int:
            """Returns the node `X` in the first edge `(X, node)`.

            Parameters
            ----------
            - node (int): the node in the computational graph for which we whish to find the input source (either another node in the
                graph or the original input itself to the network).

            Returns
            ----------
            - input source (int): this indicates the node in the computational graph providing the input to `node`. If `node` is
                receiving outside input (i.e., it is a starting node) the return will be -1. For example, this will be the case
                when a network with two independent branches (each starts from a different "input node") merge along the computational graph.
            """
            for edge in edges_list:
                if edge[1] == node:
                    #   TODO nodes originally receiving input from merge will appear twice in the list of edges, one
                    # edge per input to the merge layer. For now both inputs to a `Merge` have the same dimensions
                    # necessarily so this works for now but later will have to be revised.
                    return edge[0]
            return -1

        # access the I/O shapes for each node in `self._sinabs_edges` from the original graph in `self._graph_extractor`.
        for dcnnl_idx, dcnnl_data in self._nodes_to_dcnnl_map.items():
            for node, node_data in dcnnl_data.items():
                # node dictionary with layer data.
                if isinstance(node, int):
                    _in, _out = self._graph_extractor.get_node_io_shapes(node)

                    # update node I/O shape in the mapper (drop batch dimension).
                    if node != 0:
                        #   Find node outputing into the current node being processed (this will be the input shape). This is
                        # necessary cuz if a node originally receives input from a `nn.Flatten` for instance, when mapped into
                        # a `DynapcnnLayer` it will be receiving the input from a privious `sl.SumPool2d`.
                        input_node = find_my_input(self._sinabs_edges, node)

                        if input_node == -1:
                            # node does not have an input source within the graph (it consumes the original input to the model).
                            node_data["input_shape"] = tuple(list(_in)[1:])
                        else:
                            # input comes from another node in the graph.
                            _, _input_source_shape = (
                                self._graph_extractor.get_node_io_shapes(input_node)
                            )
                            node_data["input_shape"] = tuple(
                                list(_input_source_shape)[1:]
                            )
                    else:
                        # first node does not have an input source within the graph.
                        node_data["input_shape"] = tuple(list(_in)[1:])

                    node_data["output_shape"] = tuple(list(_out)[1:])

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
