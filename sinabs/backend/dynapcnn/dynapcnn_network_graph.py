"""
functionality : extracts the computational graph of a network defined as a `nn.Module` and converts it into a set of `DynapcnnLayer`s
                that implement a network ()`DynapcnnNetwork`) instance that can be deployed to a Speck chip.
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

import time
from typing import List, Optional, Sequence, Tuple, Union, Dict

import samna
import sinabs.layers as sl
import torch
import torch.nn as nn

import sinabs

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .io import open_device
from .utils import (
    DEFAULT_IGNORED_LAYER_TYPES,
    build_from_graph,
    build_nodes_to_dcnnl_map,
    parse_device_id,
)

from .NIRGraphExtractor import NIRtoDynapcnnNetworkGraph
from .sinabs_edges_handler import merge_handler

from .dynapcnnnetwork_module import DynapcnnNetworkModule

class DynapcnnNetworkGraph(nn.Module):
    def __init__(
        self,
        snn: nn.Module,
        input_shape: Tuple[int, int, int],
        dvs_input: bool = False,
        discretize: bool = True
    ):
        """
            Given a sinabs spiking network, prepare a dynapcnn-compatible network. This can be used to
        test the network will be equivalent once on DYNAPCNN. This class also provides utilities to
        make the dynapcnn configuration and upload it to DYNAPCNN.

        Some of the properties defined within the class constructor are meant to be temporary data structures handling the conversion
        of the `snn` (the original `nn.Module`) into a set of `DynapcnnLayer`s composing a `DynapcnnNetwork` instance. Once their role
        in preprocessing `snn` is finished, all required data to train/deploy the `DynapcnnNetwork` instance is within `self._dcnnl_edges`
        (the connectivity between each `DynapcnnLayer`/core), `self._forward_map` (every `DynapcnnLayer` in the network) and `self._merge_points`
        (the `DynapcnnLayer`s that need a `Merge` input). Thus, the following private properties are delted as last step of the constructor:

        - self._graph_tracer
        - self._sinabs_edges
        - self._sinabs_modules_map
        - self._nodes_name_remap
        - self._nodes_to_dcnnl_map
        - self._dynapcnn_layers

        Parameters
        ----------
            snn (nn.Module): a  implementing a spiking network.
            input_shape (tuple): a description of the input dimensions as `(features, height, width)`.
            dvs_input (bool): wether or not dynapcnn receive input from its DVS camera.
            discretize (bool): If `True`, discretize the parameters and thresholds. This is needed for uploading 
                weights to dynapcnn. Set to `False` only for testing purposes.
        """
        super().__init__()

        # TODO for now the graph part is not taking into consideration DVS inputs.
        # check if dvs input is expected.
        dvs_input = False
        self.dvs_input = dvs_input
        self.input_shape = input_shape

        assert len(self.input_shape) == 3, "infer_input_shape did not return 3-tuple"

        # computational graph from original PyTorch module.
        # TODO - bacth size must be passed as argument.
        self._graph_tracer = NIRtoDynapcnnNetworkGraph(
            snn,
            torch.randn((1, *self.input_shape)))                # needs the batch dimension.        

        self._sinabs_edges, \
            self._sinabs_modules_map, \
                    self._nodes_name_remap = self._get_sinabs_edges_and_modules()
        
        # create a dict holding the data necessary to instantiate a `DynapcnnLayer`.
        self._nodes_to_dcnnl_map = build_nodes_to_dcnnl_map(
            layers=self._sinabs_modules_map, 
            edges=self._sinabs_edges)
        
        # updates 'self._nodes_to_dcnnl_map' to include the I/O shape for each node.
        self._populate_nodes_io()

        # build `DynapcnnLayer` instances from graph edges and mapper.
        self._dynapcnn_layers = build_from_graph(
            discretize=discretize,
            edges=self._sinabs_edges,
            nodes_to_dcnnl_map=self._nodes_to_dcnnl_map)
        
        # these gather all data necessay to implement the forward method for this class.
        self._dcnnl_edges, self._forward_map, self._merge_points = self._get_network_module()

        # all necessary `DynapcnnLayer` data held in `self._forward_map`: removing intermediary data structures no longer necessary.
        del self._graph_tracer
        del self._sinabs_edges
        del self._sinabs_modules_map
        del self._nodes_name_remap
        del self._nodes_to_dcnnl_map
        del self._dynapcnn_layers

    ####################################################### Public Methods #######################################################
        
    @property
    def forward_map(self) -> dict:
        """ This dictionary contains each `DynapcnnLayer` in the model indexed by their ID (layer index). """
        return self._forward_map

    def forward(self, x):
        """ ."""

        layers_outputs = {}

        #   TODO - currently `node 0` (this 1st node in the 1st edge of `self._dcnnl_edges`) is always taken to be the
        # input node of the network. This won't work in cases where there are more the one input nodes to the network
        # so this functionality needs some refactoring.
        self._forward_map[self._dcnnl_edges[0][0]](x)

        # forward the input `x` through the input `DynapcnnLayer` in the `DynapcnnNetwork`s graph (1st node in the 1st edge in `self._dcnnl_edges`).
        layers_outputs[self._dcnnl_edges[0][0]] = self._forward_map[self._dcnnl_edges[0][0]](x)

        # propagate outputs in `layers_outputs` through the rest of the nodes of `self._dcnnl_edges`.
        for edge in self._dcnnl_edges:
            
            # target DynapcnnLayer (will consume tensors from `layers_outputs`).
            trg_dcnnl = edge[1]

            if trg_dcnnl in self._merge_points and trg_dcnnl not in layers_outputs:
                # by this points the arguments of the `Merge` associated with `trg_dcnnl` should have been computed.
                arg1, arg2 = self._merge_points[trg_dcnnl]['sources']

                #   find which returned tensor from the `forward` call of DynapcnnLayers `arg1` and `arg2` are to be fed
                # to the target DynapcnnLayer `trg_dcnnl`.
                return_index_arg1 = self._forward_map[arg1].get_destination_dcnnl_index(trg_dcnnl)
                return_index_arg2 = self._forward_map[arg2].get_destination_dcnnl_index(trg_dcnnl)

                # retrieve input tensors to `Merge`.
                _arg1 = layers_outputs[arg1][return_index_arg1]
                _arg2 = layers_outputs[arg2][return_index_arg2]

                # merge tensors.
                merge_output = self._merge_points[trg_dcnnl]['merge'](_arg1, _arg2)

                # call the forward.
                layers_outputs[trg_dcnnl] = self._forward_map[trg_dcnnl](merge_output)

            elif trg_dcnnl not in layers_outputs:
                # input source for `trg_dcnnl`.
                src_dcnnl = edge[0]

                #   find which returned tensor from the `forward` call of the source DynapcnnLayer `src_dcnnl` is to be fed
                # to the target DynapcnnLayer `trg_dcnnl`.
                return_index = self._forward_map[src_dcnnl].get_destination_dcnnl_index(trg_dcnnl)

                # call the forward.
                layers_outputs[trg_dcnnl] = self._forward_map[trg_dcnnl](layers_outputs[src_dcnnl][return_index])

            else:

                pass
        
        # TODO - this assumes the network has a single output node.
        # last computed is the output layer.
        return layers_outputs[trg_dcnnl][0]
    
    def parameters(self) -> list:
        """ Gathers all the parameters of the network in a list. This is done by accessing the convolutional layer in each `DynapcnnLayer`, calling 
        its `.parameters` method and saving it to a list.

        Note: the method assumes no biases are used.

        Returns
        ----------
            parameters (list): a list of parameters of all convolutional layers in the `DynapcnnNetwok`.
        """
        parameters = []

        for layer in self._forward_map.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                parameters.extend(layer.conv_layer.parameters())

        return parameters
    
    def init_weights(self, init_fn: nn.init = nn.init.xavier_normal_) -> None:
        """ Call the weight initialization method `init_fn` on each `DynapcnnLayer.conv_layer.weight.data` in the `DynapcnnNetwork` instance."""
        for layer in self._forward_map.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                init_fn(layer.conv_layer.weight.data)

    def detach_neuron_states(self) -> None:
        """ Detach the neuron states and activations from current computation graph (necessary). """

        for module in self._forward_map.values():
            if isinstance(module, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
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
                    dk_io.set_slow_clk_rate(slow_clk_frequency) # Hz

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

                return print(self)
            
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
        has_dvs_layer = isinstance(self._forward_map[0], DVSLayer)

        if chip_layers_ordering == "auto":
            # figure out mapping of each DynapcnnLayer into one core.
            chip_layers_ordering = config_builder.get_valid_mapping(self)

        else:
            # TODO - mapping from each DynapcnnLayer into cores has been provided by the user: NOT IMPLEMENTED YET.
            if has_dvs_layer:
                # TODO not handling DVSLayer yet.
                pass

        # update config.
        config = config_builder.build_config(self, None)

        # TODO not handling DVSLayer yet (this is from the old implementation, should be revised).
        if self.input_shape and self.input_shape[0] == 1:
            config.dvs_layer.merge = True

        # TODO all this monitoring part needs validation still.
        monitor_chip_layers = []
        if monitor_layers is None:
            # check if any monitoring is enabled (if not, enable monitoring for the last layer).
            for dcnnl_index, ith_dcnnl in self._forward_map.items():
                if len(ith_dcnnl.dynapcnnlayer_destination) == 0:
                    monitor_chip_layers.append(ith_dcnnl.assigned_core)
                    break
        elif monitor_layers == "all":
            for dcnnl_index, ith_dcnnl in self._forward_map.items():
                # TODO not handling DVSLayer yet
                # monitor each chip core (if not a DVSLayer).
                if not isinstance(ith_dcnnl, DVSLayer):
                    monitor_chip_layers.append(ith_dcnnl.assigned_core)
        
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
        """ Uses the `DynapcnnLayer` instances in `self._dynapcnn_layers` and the connectivity between them to create three data structures 
        that guide the data forwarding between the layer during the forward pass.

        Note: the property `DynapcnnLayer.assigned_core` is only set after `self.to(device='speck...')` is called.

        Returns
        ----------
            dcnnl_edges (list): edges, represented as tuples of `DynapcnnLayer` indices, used to guide the data forwarding through each `DynapcnnLayer` in forward method.
            forward_map (dict): have all the `DynapcnnLayer` (`value`), each being accessible via its index (`key`). Used to call `DynapcnnLayer.forward` in forward method.
            merge_points (dict): used to compose the inputs to a `DynapcnnLayer` that requires an input from a `Merge` layer.
        """

        # get connections between `DynapcnnLayer`s.
        dcnnl_edges = self._get_dynapcnnlayers_edges()

        dcnnnet_module = DynapcnnNetworkModule(dcnnl_edges, self._dynapcnn_layers)

        return dcnnnet_module.dcnnl_edges, dcnnnet_module.forward_map, dcnnnet_module.merge_points
    
    def _get_dynapcnnlayers_edges(self) -> List[Tuple[int, int]]:
        """ Create edges representing connections between `DynapcnnLayer` instances. """
        dcnnl_edges = []

        for dcnnl_idx, layer_data in self._dynapcnn_layers.items():
            for dest in layer_data['destinations']:
                dcnnl_edges.append((dcnnl_idx, dest))
        
        return dcnnl_edges
    
    def _get_sinabs_edges_and_modules(self) -> Tuple[List[Tuple[int, int]], Dict[int, nn.Module], Dict[int, int]]:
        """ The computational graph extracted from `snn` might contain layers that are ignored (e.g. a `nn.Flatten` will be
        ignored when creating a `DynapcnnLayer` instance). Thus the list of edges from such model need to be rebuilt such that if there are
        edges `(A, X)` and `(X, B)`, and `X` is an ignored layer, an edge `(A, B)` is created.

        Returns
        ----------
            edges_without_merge (list): a list of edges based on `sinabs_edges` but where edges involving a `Merge` layer have been 
                remapped to connect the nodes involved in the merging directly.
            sinabs_modules_map (dict): a dict containing the nodes of the graph (described now by `edges_without_merge`) as `key` and 
                their associated module as `value`.
            remapped_nodes (dict): a dict where `key` is the original node name (as extracted by `self._graph_tracer`) and `value` is
                the new node name (after ignored layers have been dropped and `Merge` layers have be processed before being removed).
        """
        
        # remap `(A, X)` and `(X, B)` into `(A, B)` if `X` is a layer in the original `snn` to be ignored.
        sinabs_edges, remapped_nodes = self._graph_tracer.remove_ignored_nodes( 
            DEFAULT_IGNORED_LAYER_TYPES)

        # nodes (layers' "names") need remapping in case some layers have been removed (e.g. a `nn.Flattern` is ignored).
        sinabs_modules_map = {}
        for orig_name, new_name in remapped_nodes.items():
            sinabs_modules_map[new_name] = self._graph_tracer.modules_map[orig_name]

        # bypass merging layers to connect the nodes involved in them directly to the node where the merge happens.
        edges_without_merge = merge_handler(sinabs_edges, sinabs_modules_map)

        return edges_without_merge, sinabs_modules_map, remapped_nodes
    
    def _populate_nodes_io(self):
        """ Loops through the nodes in the original graph to retrieve their I/O tensor shapes and add them to their respective
        representations in `self._nodes_to_dcnnl_map`."""

        def find_original_node_name(name_mapper: dict, node: int):
            """ Find what a node is originally named when built in `self._graph_tracer`. """
            for orig_name, new_name in name_mapper.items():
                if new_name == node:
                    return orig_name
            raise ValueError(f'Node {node} could not be found within the name remapping done by self._get_sinabs_edges_and_modules().')
        
        def find_my_input(edges_list: list, node: int) -> int:
            """ Returns the node `X` in the first edge `(X, node)`.

            Parameters
            ----------
                node (int): the node in the computational graph for which we whish to find the input source (either another node in the
                    graph or the original input itself to the network).
            
            Returns
            ----------
                input source (int): this indicates the node in the computational graph providing the input to `node`. If `node` is
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

        # access the I/O shapes for each node in `self._sinabs_edges` from the original graph in `self._graph_tracer`.
        for dcnnl_idx, dcnnl_data in self._nodes_to_dcnnl_map.items():
            for node, node_data in dcnnl_data.items():
                # node dictionary with layer data.
                if isinstance(node, int):
                    # some nodes might have been renamed (e.g. after droppping a `nn.Flatten`), so find how node was originally named.
                    orig_name = find_original_node_name(self._nodes_name_remap, node)
                    _in, _out = self._graph_tracer.get_node_io_shapes(orig_name)

                    # update node I/O shape in the mapper (drop batch dimension).
                    if node != 0:
                        #   Find node outputing into the current node being processed (this will be the input shape). This is
                        # necessary cuz if a node originally receives input from a `nn.Flatten` for instance, when mapped into
                        # a `DynapcnnLayer` it will be receiving the input from a privious `sl.SumPool2d`.
                        input_node = find_my_input(self._sinabs_edges, node)

                        if input_node == -1:
                            # node does not have an input source within the graph (it consumes the original input to the model).
                            node_data['input_shape'] = tuple(list(_in)[1:])
                        else:
                            # input comes from another node in the graph.
                            input_node_orig_name = find_original_node_name(self._nodes_name_remap, input_node)
                            _, _input_source_shape = self._graph_tracer.get_node_io_shapes(input_node_orig_name)
                            node_data['input_shape'] = tuple(list(_input_source_shape)[1:])
                    else:
                        # first node does not have an input source within the graph.
                        node_data['input_shape'] = tuple(list(_in)[1:])

                    node_data['output_shape'] = tuple(list(_out)[1:])

    def _to_device(self, device: torch.device) -> None:
        """ ."""
        for layer in self._forward_map.values():
            if isinstance(layer, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                layer.conv_layer.to(device)
                layer.spk_layer.to(device)
                
                # if there's more than one pooling each of them becomes a node that is catched by the `else` statement.
                if len(layer.pool_layer) == 1:
                    layer.pool_layer[0].to(device)
            else:
                # this nodes are created from `DynapcnnLayer`s that have multiple poolings (each pooling becomes a new node).
                layer.to(device)

    def __str__(self):
        pretty_print = ''
        for idx, layer_data in self._forward_map.items():
            pretty_print += f'----------------------- [ DynapcnnLayer {idx} ] -----------------------\n'
            pretty_print += f'{layer_data}\n\n'
                
        return pretty_print