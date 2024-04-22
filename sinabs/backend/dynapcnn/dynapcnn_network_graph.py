# functionality : ...
# author        : Willian Soares Girao
# contact       : williansoaresgirao@gmail.com

import time
from subprocess import CalledProcessError
from typing import List, Optional, Sequence, Tuple, Union, Dict

import samna
import sinabs.layers
import torch
import torch.nn as nn

import sinabs

from .chip_factory import ChipFactory
from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .io import disable_timestamps, enable_timestamps, open_device, reset_timestamps
from .utils import (
    DEFAULT_IGNORED_LAYER_TYPES,
    build_from_graph,
    build_nodes_to_dcnnl_map,
    parse_device_id,
)

from .graph_tracer import GraphTracer
from .exceptions import InvalidTorchModel
from warnings import warn

from .NIRGraphExtractor import NIRtoDynapcnnNetworkGraph
from .sinabs_edges_handler import merge_handler

class DynapcnnNetworkGraph(nn.Module):
    """Given a sinabs spiking network, prepare a dynapcnn-compatible network. This can be used to
    test the network will be equivalent once on DYNAPCNN. This class also provides utilities to
    make the dynapcnn configuration and upload it to DYNAPCNN.
    """

    def __init__(
        self,
        snn: Union[nn.Sequential, sinabs.Network, nn.Module],
        input_shape: Optional[Tuple[int, int, int]] = None,
        dvs_input: bool = False,
        discretize: bool = True
    ):
        """
        DynapcnnNetworkGraph: a class turning sinabs networks into dynapcnn
        compatible networks, and making dynapcnn configurations.

        Parameters
        ----------
            snn: ...
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

        # TODO for now the graph part is not taking into consideration this.
        # check if dvs input is expected.
        dvs_input = False
        self.dvs_input = dvs_input
        self.input_shape = input_shape

        assert len(self.input_shape) == 3, "infer_input_shape did not return 3-tuple"

        # computational graph from original PyTorch module.
        self.graph_tracer = NIRtoDynapcnnNetworkGraph(
            snn,
            torch.randn((1, *self.input_shape)))                # needs the batch dimension.        

        self.sinabs_edges, \
            self.sinabs_modules_map, \
                    self.nodes_name_remap = self.get_sinabs_edges_and_modules()
        
        self.nodes_to_dcnnl_map = build_nodes_to_dcnnl_map(
            layers=self.sinabs_modules_map, 
            edges=self.sinabs_edges)
        
        # update the I/O shapes for each layer in 'self.nodes_to_dcnnl_map'.
        self.populate_nodes_io()

        # # build model from graph edges.
        # self.dynapcnn_layers = build_from_graph(
        #     discretize=discretize,
        #     edges=self.sinabs_edges,
        #     nodes_to_dcnnl_map=self.nodes_to_dcnnl_map)
        
        # for dcnnl_idx, dcnnl_data in self.nodes_to_dcnnl_map.items():
        #     print(f'DynapcnnLayer-index {dcnnl_idx}')
        #     for key, val in dcnnl_data.items():
        #         if isinstance(key, int):
        #             print(key, val['layer'], val['input_shape'])
        #         else:
        #             print(key, val)
        #     print('\n')

    def __str__(self):
        pretty_print = ''
        for idx, layer_data in self.dynapcnn_layers.items():
            layer = layer_data['layer']
            dest = layer_data['destinations']
            core = layer_data['core_idx']
            pretty_print += f'\nlayer index: {idx}\nlayer modules: {layer}\nlayer destinations: {dest}\nassigned core: {core}\n'
        return pretty_print
    
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

            if device_name in ChipFactory.supported_devices:                # pragma: no cover
                
                config = self.make_config(                                  # generate config.
                    chip_layers_ordering=chip_layers_ordering,
                    device=device,
                    monitor_layers=monitor_layers,
                    config_modifier=config_modifier,
                )

                self.samna_device = open_device(device)                     # apply configuration to device.
                self.samna_device.get_model().apply_configuration(config)
                time.sleep(1)

                if slow_clk_frequency is not None:                          # set external slow-clock if needed.
                    dk_io = self.samna_device.get_io_module()
                    dk_io.set_slow_clk(True)
                    dk_io.set_slow_clk_rate(slow_clk_frequency)             # Hz

                builder = ChipFactory(device).get_config_builder()
                
                self.samna_input_buffer = builder.get_input_buffer()        # create input source node.
                self.samna_output_buffer = builder.get_output_buffer()      # create output sink node node.

                self.device_input_graph = samna.graph.EventFilterGraph()    # connect source node to device sink.
                self.device_input_graph.sequential(
                    [
                        self.samna_input_buffer,
                        self.samna_device.get_model().get_sink_node(),
                    ]
                )

                self.device_output_graph = samna.graph.EventFilterGraph()   # connect sink node to device.
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

        if is_compatible:                                           # validate config.
            print("Network is valid")
            return config
        else:
            raise ValueError(f"Generated config is not valid for {device}")
        
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

        has_dvs_layer = isinstance(self.dynapcnn_layers[0]['layer'], DVSLayer)

        if chip_layers_ordering == "auto":                           # figure out mapping of each DynapcnnLayer into one core.
            chip_layers_ordering = config_builder.get_valid_mapping(self)

        else:                                                        # mapping from each DynapcnnLayer into cores has been provided.
            if has_dvs_layer:
                pass                                                 # TODO not handling DVSLayer yet.

        config = config_builder.build_config(self, None)             # update config.

        if self.input_shape and self.input_shape[0] == 1:            # ???
            config.dvs_layer.merge = True

        monitor_chip_layers = []                                     # TODO all this monitoring part needs validation still.
        if monitor_layers is None:                                   # check if any monitoring is enabled (if not, enable monitoring for the last layer).
            for _, dcnnl_data in self.dynapcnn_layers.items():
                if len(dcnnl_data['destinations']) == 0:
                    monitor_chip_layers.append(dcnnl_data['core_idx'])
                    break
        elif monitor_layers == "all":
            for _, dcnnl_data in self.dynapcnn_layers.items():      # monitor each chip core (if not a DVSLayer).
                if not isinstance(dcnnl_data['layer'], DVSLayer):
                    monitor_chip_layers.append(dcnnl_data['core_idx'])
        
        if monitor_layers:
            if "dvs" in monitor_layers:
                monitor_chip_layers.append("dvs")

        config_builder.monitor_layers(config, monitor_chip_layers)   # enable monitors on the specified layers.

        if config_modifier is not None:                              # apply user config modifier.
            config = config_modifier(config)

        return config, config_builder.validate_configuration(config) # validate config.
    
    def get_sinabs_edges_and_modules(self) -> Tuple[List[Tuple[int, int]], Dict[int, nn.Module]]:
        """ The computational graph extracted from `snn` might contain layers that are ignored (e.g. a `nn.Flatten` will be
        ignored when creating a `DynapcnnLayer` instance). Thus the list of edges from such model need to be rebuilt such that if there are
        edges `(A, X)` and `(X, B)`, and `X` is an ignored layer, an edge `(A, B)` is created.

        Returns
        ----------
            sinabs_edges: a list of tuples representing the edges between the layers of a sinabs model.
            sinabs_modules_map: a dictionary containing the nodes of the graph as `key` and their associated module as `value`.
        """
        sinabs_edges, remapped_nodes = self.graph_tracer.remove_ignored_nodes(  # remap `(A, X)` and `(X, B)` into `(A, B)`.
            DEFAULT_IGNORED_LAYER_TYPES)

        sinabs_modules_map = {}                                                 # nodes (layers' "names") need remapping in case some layers have been removed (e.g. nn.Flattern()).
        for orig_name, new_name in remapped_nodes.items():
            sinabs_modules_map[new_name] = self.graph_tracer.modules_map[orig_name]

        edges_without_merge, merge_nodes = merge_handler(sinabs_edges, sinabs_modules_map)   # bypass merging layers to connect the nodes involved in them directly to the node where the merge happens.

        return edges_without_merge, sinabs_modules_map, remapped_nodes
    
    def populate_nodes_io(self):
        """ ."""

        for edge in self.sinabs_edges:
            print(edge)
        print('\n')

        def find_original_node_name(name_mapper, node):
            for orig_name, new_name in name_mapper.items():
                if new_name == node:
                    return orig_name
            raise ValueError(f'Node {node} could not be found within the name remapping done by self.get_sinabs_edges_and_modules().')
        
        def find_my_input(edges_list, node):
            for edge in edges_list:
                if edge[1] == node:
                    #   TODO nodes originally receiving input from merge will appear twice in the list of edges, one
                    # edge per input to the merge layer. For now both inputs to a `Merge` have the same dimensions 
                    # necessarily so this works for now but later will have to be revised.
                    return edge[0]
            raise ValueError(f'Node {node} is not receiving input from any other node in the graph.')

        # access the I/O shapes for each node in `self.sinabs_edges` from the original graph in `self.graph_tracer`.
        for dcnnl_idx, dcnnl_data in self.nodes_to_dcnnl_map.items():
            for node, node_data in dcnnl_data.items():
                # node dictionary with layer data.
                if isinstance(node, int):
                    # some nodes might have been renamed (e.g. after droppping a `nn.Flatten`), so find how node was originally named.
                    orig_name = find_original_node_name(self.nodes_name_remap, node)
                    _in, _out = self.graph_tracer.get_node_io_shapes(orig_name)

                    # update node I/O shape in the mapper (drop batch dimension).
                    if node != 0:
                        #   Find node outputing into the current node being processed (this will be the input shape). This is
                        # necessary cuz if a node originally receives input from a `nn.Flatten` for instance, when mapped into
                        # a `DynapcnnLayer` it will be receiving the input from a privious `sl.SumPool2d`.
                        input_node = find_my_input(self.sinabs_edges, node)
                        input_node_orig_name = find_original_node_name(self.nodes_name_remap, input_node)
                        _, _input_source_shape = self.graph_tracer.get_node_io_shapes(input_node_orig_name)
                        node_data['input_shape'] = tuple(list(_input_source_shape)[1:])
                    else:
                        # first node does not have an input source within the graph.
                        node_data['input_shape'] = tuple(list(_in)[1:])

                    node_data['output_shape'] = tuple(list(_out)[1:])

        for dcnnl_idx, dcnnl_data in self.nodes_to_dcnnl_map.items():
            print(f'DynapcnnLayer-index {dcnnl_idx}')
            for key, val in dcnnl_data.items():
                if isinstance(key, int):
                    print(key, val['input_shape'], val['output_shape'])
            print('\n')