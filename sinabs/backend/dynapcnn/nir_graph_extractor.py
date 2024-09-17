# author    : Willian Soares Girao
# contact   : wsoaresgirao@gmail.com

import copy
from typing import Tuple, Dict, List

import nirtorch
import sinabs
import torch
import torch.nn as nn

from .utils import topological_sorting

class NIRtoDynapcnnNetworkGraph():
    def __init__(self, spiking_model: nn.Module, dummy_input: torch.tensor):
        """ Class implementing the extraction of the computational graph from `spiking_model`, where
        each node represents a layer in the model and the list of edges represents how the data flow between
        the layers. 

        Parameters
        ----------
        - spiking_model (nn.Module): a sinabs-compatible spiking network.
        - dummy_input (torch.tensor): a random input sample to be fed through the model to acquire both
            the computational graph (via `nirtorch`) and the I/O shapes of each node. Its a 4-D shape
            with `(batch, channels, heigh, width)`.

        Attributes
        ----------
        - edges_list (list of 2-tuples of integers):
            Tuples describing the connections between layers in `spiking_model`.
            Each layer (node) is identified by a unique integer ID.
        - name_2_index_map (dict):
            Keys are original variable names of layers in `spiking_model`.
            Values are unique integer IDs.
        - entry_nodes (list of ints):
            IDs of nodes acting as entry points for the network, i.e. receiving external input.
        - modules_map (dict):
            Map from layer ID to the corresponding nn.Module instance.
        """

        # extract computational graph.
        nir_graph = nirtorch.extract_torch_graph(spiking_model, dummy_input, model_name=None).ignore_tensors()

        # converts the NIR representation into a list of edges with nodes represented as integers.
        self._edges_list, self._name_2_indx_map, self._entry_nodes = self._get_edges_from_nir(nir_graph)
        
        # recovers the associated `nn.Module` (layer) of each node.
        self.modules_map = self._get_named_modules(spiking_model)

        # retrieves what the I/O shape for each node's module is.
        self._nodes_io_shapes = self._get_nodes_io_shapes(dummy_input)

    ####################################################### Publich Methods #######################################################

    @property
    def entry_nodes(self) -> List[int]:
        return self._entry_nodes

    @property
    def edges_list(self):
        return self._edges_list
    
    @property
    def name_2_indx_map(self):
        return self._name_2_indx_map
    
    @property
    def nodes_io_shapes(self):
        return self._nodes_io_shapes

    def remove_ignored_nodes(self, default_ignored_nodes: tuple) -> Tuple[list, dict]:
        """ Recreates the edges list based on layers that `DynapcnnNetwork` will ignore. This
        is done by setting the source (target) node of an edge where the source (target) node
        will be dropped as the node that originally targeted this node to be dropped.

        Parameters
        ----------
        - default_ignored_nodes (tuple): a set of layers (`nn.Module`) that should be ignored from the graph.

        Returns
        ----------
        - remapped_edges (list): the new list of edges after nodes flagged by `default_ignored_nodes` have been removed.
        - remapped_nodes (dict): updated nodes' IDs after nodes flagged by `default_ignored_nodes` have been removed.
        """
        edges = copy.deepcopy(self._edges_list)
        parsed_edges = []
        removed_nodes = []

        # removing ignored nodes from edges.
        for edge_idx in range(len(edges)):
            _src = edges[edge_idx][0]
            _trg = edges[edge_idx][1]

            if isinstance(self.modules_map[_src], default_ignored_nodes):
                removed_nodes.append(_src)
                # all edges where node '_src' is target change it to node '_trg' as their target.
                for edge in edges:
                    if edge[1] == _src:
                        new_edge = (edge[0], _trg)
            elif isinstance(self.modules_map[_trg], default_ignored_nodes):
                removed_nodes.append(_trg)
                # all edges where node '_trg' is source change it to node '_src' as their source.
                for edge in edges:
                    if edge[0] == _trg:
                        new_edge = (_src, edge[1])
            else:
                new_edge = (_src, _trg)
            
            if new_edge not in parsed_edges:
                parsed_edges.append(new_edge)

        removed_nodes = list(set(removed_nodes))

        # remapping nodes indexes.
        remapped_nodes = {}
        for node_indx, __ in self.modules_map.items():
            _ = [x for x in removed_nodes if node_indx > x]
            remapped_nodes[node_indx] = node_indx - len(_)
            
        for x in removed_nodes:
            del remapped_nodes[x]

        # remapping nodes names in parsed edges.
        remapped_edges = []
        for edge in parsed_edges:
            remapped_edges.append((remapped_nodes[edge[0]], remapped_nodes[edge[1]]))

        return remapped_edges, remapped_nodes
    
    # TODO - it would be good if I/O shapes were returned by the NIR graph.
    def get_node_io_shapes(self, node: int) -> Tuple[torch.Size, torch.Size]:
        """ Returns the I/O tensors' shapes of `node`.

        Returns
        ----------
        - input shape (torch.Size): shape of the input tensor to `node`.
        - output shape (torch.Size): shape of the output tensor from `node`.
        """
        return self._nodes_io_shapes[node]['input'], self._nodes_io_shapes[node]['output']

    ####################################################### Pivate Methods #######################################################

    def _get_edges_from_nir(self, nir_graph: nirtorch.graph.Graph) -> Tuple[List[Tuple[int, int]], Dict[str, int], List[int]]:
        """ Standardize the representation of `nirtorch.graph.Graph` into a list of edges (`Tuple[int, int]`) where
        each node in `nir_graph` is represented by an interger (with the source node starting as `0`).

        Parameters
        ----------
        - nir_graph (nirtorch.graph.Graph): a NIR graph representation of `spiking_model`.
        
        Returns
        ----------
        - edges_list (list): tuples describing the connections between layers in `spiking_model`.
        - name_2_indx_map (dict): `key` is the original variable name for a layer in `spiking_model` and `value
            is an integer representing the layer in a standard format.
        - entry_nodes (list): IDs of nodes acting as entry points for the network (i.e., receiving external input).
        """
        edges_list = []
        name_2_indx_map = {}
        idx_counter = 0                                         # TODO maybe make sure the input node from nir always gets assined `0`.

        nodes_IDs = [0]

        for src_node in nir_graph.node_list:
            # source node.
            if src_node.name not in name_2_indx_map:
                name_2_indx_map[src_node.name] = idx_counter
                idx_counter += 1

                nodes_IDs.append(idx_counter)

            for trg_node in src_node.outgoing_nodes:
                # target node.
                if trg_node.name not in name_2_indx_map:
                    name_2_indx_map[trg_node.name] = idx_counter
                    idx_counter += 1

                    nodes_IDs.append(idx_counter)

                edges_list.append((name_2_indx_map[src_node.name], name_2_indx_map[trg_node.name]))

        # finding entry/exits nodes of the graph.
        all_sources = [x[0] for x in edges_list]
        all_targets = [x[1] for x in edges_list]

        entry_nodes = list(set(all_sources) - set(all_targets))

        return edges_list, name_2_indx_map, entry_nodes
    
    def _get_named_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        """ Find for each node in the graph what its associated layer in `model` is.

        Parameters
        ----------
        - model (nn.Module): the `spiking_model` used as argument to the class instance.

        Returns
        ----------
        - modules_map (dict): the mapping between a node (`key` as an `int`) and its module (`value` as a `nn.Module`).
        """
        modules_map = {}

        if isinstance(model, nn.Sequential):                    # TODO shouldn't accept `nn.Sequential` any longer.
            # access modules via `.named_modules()`.
            for name, module in model.named_modules():
                if name != '':
                    # skip the module itself.
                    modules_map[self._name_2_indx_map[name]] = module

        elif isinstance(model, nn.Module):
            # access modules via `.named_children()`.
            for name, module in model.named_children():
                modules_map[self._name_2_indx_map[name]] = module

        else:
            raise ValueError('Either a nn.Sequential or a nn.Module is required.')

        return modules_map
    
    def _get_nodes_io_shapes(self, input_dummy: torch.tensor) -> Dict[int, Dict[str, torch.Size]]:
        """ Iteratively calls the forward method of each `nn.Module` (i.e., a layer/node in the graph) using the topologically
        sorted nodes extracted from the computational graph of the model being parsed.

        Parameters
        ----------
        - input_dummy (torch.tensor): a sample (random) tensor of the sort of input being fed to the network.

        Returns
        ----------
        - nodes_io_map (dict): a dictionary mapping nodes to their I/O shapes.
        """
        nodes_io_map = {}

        # topological sorting of the graph.
        temp_edges_list = copy.deepcopy(self._edges_list)
        for node in self._entry_nodes:
            temp_edges_list.append(('input', node))
        sorted_nodes = topological_sorting(temp_edges_list)

        # propagate inputs through the nodes.
        for node in sorted_nodes:

            if isinstance(self.modules_map[node], sinabs.layers.merge.Merge):
                # find `Merge` arguments (at this point the output of Merge has to have been calculated).
                arg1, arg2 = self._find_merge_arguments(node)

                # retrieve arguments output tensors.
                arg1_out = nodes_io_map[arg1]['output']
                arg2_out = nodes_io_map[arg2]['output']

                # TODO - this is currently a limitation inpused by the validation checks done by Speck once a configuration: it wants two 
                # different input sources to a core to have the same output shapes.
                if arg1_out.shape != arg2_out.shape:
                    raise ValueError(f'Layer `sinabs.layers.merge.Merge` (node {node}) require two input tensors with the same shape: arg1.shape {arg1_out.shape} differs from arg2.shape {arg2_out.shape}.')

                # forward input through the node.
                _output = self.modules_map[node](arg1_out, arg2_out)

                # save node's I/O tensors.
                nodes_io_map[node] = {'input': arg1_out, 'output': _output}

            else:

                if node in self._entry_nodes:
                    # forward input dummy through node.
                    _output = self.modules_map[node](input_dummy)

                    # save node's I/O tensors.
                    nodes_io_map[node] = {'input': input_dummy, 'output': _output}

                else:
                    # find node generating the input to be used.
                    input_node = self._find_source_of_input_to(node)
                    _input = nodes_io_map[input_node]['output']

                    # forward input through the node.
                    _output = self.modules_map[node](_input)

                    # save node's I/O tensors.
                    nodes_io_map[node] = {'input': _input, 'output': _output}

        # replace the I/O tensor information by its shape information.
        for node, io in nodes_io_map.items():
            nodes_io_map[node]['input'] = io['input'].shape
            nodes_io_map[node]['output'] = io['output'].shape

        return nodes_io_map

    def _find_source_of_input_to(self, node: int) -> int:
        """ Finds the first edge `(X, node)` returns `X`.

        Parameters
        ----------
        - node (int): the node in the computational graph for which we whish to find the input source (either another node in the
            graph or the original input itself to the network).
        
        Returns
        ----------
        - input source (int): ID of the node in the computational graph providing the input to `node`. If `node` is
            receiving outside input (i.e., it is a starting node) the return will be -1. For example, this will be the case 
            when a network with two independent branches (each starts from a different "input node") merge along the computational graph.
        """
        for edge in self._edges_list:
            if edge[1] == node:
                return edge[0]

        return -1

    def _find_merge_arguments(self, merge_node: int) -> Tuple[int, int]:
        """ A `Merge` layer receives two inputs. Return the two inputs to `merge_node` representing a `Merge` layer.

        Returns
        ----------
        - args (tuple): the IDs of the nodes that provice the input arguments to a `Merge` layer.
        """
        args = []

        for edge in self._edges_list:
            if edge[1] == merge_node:
                args.append(edge[0])
        
        if len(args) == 2:
            return tuple(args)
        else:
            raise ValueError(f'Number of arguments found for `Merge` node {merge_node} is {len(args)} (should be 2).')