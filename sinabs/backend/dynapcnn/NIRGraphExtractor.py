import torch
import torch.nn as nn
import nirtorch
import copy
import sinabs
from typing import Tuple, Dict, List, Union

class NIRtoDynapcnnNetworkGraph():
    def __init__(self, spiking_model: nn.Module, dummy_input: torch.tensor):
        """ Class implementing the extraction of the computational graph from `spiking_model`, where
        each node represents a layer in the model and the list of edges represents how the data flow between
        the layers. 

        Parameters
        ----------
            spiking_model (nn.Module): a sinabs-compatible spiking network.
            dummy_input (torch.tensor): a random input sample to be fed through the model to acquire both
                the computational graph (via `nirtorch`) and the I/O shapes of each node. Its a 4-D shape
                with `(batch, channels, heigh, width)`.
        """

        # extract computational graph.
        nir_graph = nirtorch.extract_torch_graph(spiking_model, dummy_input, model_name=None).ignore_tensors()

        # converts the NIR representation into a list of edges with nodes represented as integers.
        self._edges_list, self._name_2_indx_map = self._get_edges_from_nir(nir_graph)

        for key, val in self._name_2_indx_map.items():
            print(key, val)
        print('---------------------------------------------------')
        for edge in self._edges_list:
            print(edge)
        print('---------------------------------------------------')
        
        # recovers the associated `nn.Module` (layer) of each node.
        self.modules_map = self._get_named_modules(spiking_model)

        # retrieves what the I/O shape for each node's module is.
        self._nodes_io_shapes, self._flagged_input_nodes = self._get_nodes_io_shapes(dummy_input)

    ### Publich Methods ###

    @property
    def flagged_input_nodes(self) -> List[int]:
        return self._flagged_input_nodes

    def get_edges_list(self):
        return self._edges_list

    def remove_ignored_nodes(self, default_ignored_nodes):
        """ Recreates the edges list based on layers that 'DynapcnnNetwork' will ignore. This
        is done by setting the source (target) node of an edge where the source (target) node
        will be dropped as the node that originally targeted this node to be dropped.
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
    
    def get_node_io_shapes(self, node: int) -> Tuple[torch.Size, torch.Size]:
        """ Returns the I/O tensors' shapes of `node`. """
        return self._nodes_io_shapes[node]['input'], self._nodes_io_shapes[node]['output']

    ### Pivate Methods ###

    def _get_edges_from_nir(self, nir_graph: nirtorch.graph.Graph) -> Union[List[Tuple[int, int]], Dict[str, int]]:
        """ Standardize the representation of `nirtorch.graph.Graph` into a list of edges (`Tuple[int, int]`) where
        each node in `nir_graph` is represented by an interger (with the source node starting as `0`).

        Parameters
        ----------
            nir_graph (nirtorch.graph.Graph): a NIR graph representation of `spiking_model`.
        
        Returns
        ----------
            edges_list (list): tuples describing the connections between layers in `spiking_model`.
            name_2_indx_map (dict): `key` is the original variable name for a layer in `spiking_model` and `value
                is an integer representing the layer in a standard format.
        """
        edges_list = []
        name_2_indx_map = {}
        idx_counter = 0                                         # TODO maybe make sure the input node from nir always gets assined `0`.

        for src_node in nir_graph.node_list:                    # source node.
            if src_node.name not in name_2_indx_map:
                name_2_indx_map[src_node.name] = idx_counter
                idx_counter += 1

            for trg_node in src_node.outgoing_nodes:            # target node.
                if trg_node.name not in name_2_indx_map:
                    name_2_indx_map[trg_node.name] = idx_counter
                    idx_counter += 1

                edges_list.append((name_2_indx_map[src_node.name], name_2_indx_map[trg_node.name]))

        return edges_list, name_2_indx_map
    
    def _get_named_modules(self, model: nn.Module) -> Dict[int, nn.Module]:
        """ Find for each node in the graph what its associated layer in `model` is.

        Parameters
        ----------
            model (nn.Module): the `spiking_model` used as argument to the class instance.

        Returns
        ----------
            modules_map (dict): the mapping between a node (`key` as an `int`) and its module (`value` as a `nn.Module`).
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
    
    # TODO - THIS ALSO NEEDS TOPOLOGICAL SORTING TO CORRECTLY GET I/O SHAPES UNDER ALL CIRCUNSTANCES.
    def _get_nodes_io_shapes(self, input_dummy: torch.tensor) -> Tuple[Dict[int, Dict[str, torch.Size]], List]:
        """ Loops through the graph represented in `self._edges_list` and propagates the inputs through the nodes, starting from
        `node 0` fed `input_dummy`.

        Parameters
        ----------
        - input_dummy (torch.tensor): a sample (random) tensor of the sort of input being fed to the network.

        Returns
        ----------
        - nodes_io_map (dict): a dictionary mapping nodes to their I/O shapes.
        - flagged_input_nodes (list): IDs of nodes that are receiving as input `input_dummy` (i.e., input nodes of the network).
        """
        nodes_io_map = {}
        flagged_merge_nodes = {}
        flagged_input_nodes = []

        # propagate inputs through the nodes.
        for edge in self._edges_list:
            src = edge[0]
            trg = edge[1]

            print('> ', edge)

            if isinstance(self.modules_map[src], sinabs.layers.merge.Merge):
                # At this point the output of Merge has to have been calculated.
                
                # pass input through target.
                if trg not in nodes_io_map:
                    nodes_io_map[trg] = {'input': None, 'output': None}

                    # find node generating the input to be used.
                    inp_node = self._find_source_of_input_to(trg)
                    _input = nodes_io_map[inp_node]['output']

                    # forward input through the node.
                    _output = self.modules_map[trg](_input)

                    # save node's input/output.
                    nodes_io_map[trg] = {'input': _input, 'output': _output}

            elif isinstance(self.modules_map[trg], sinabs.layers.merge.Merge):
                # Merge requires two inputs: need to check if both of its inputs have been calculated.
                if trg not in flagged_merge_nodes:
                    flagged_merge_nodes[trg] = {}
                
                args = self._find_merge_arguments(trg)

                for arg in args:
                    if arg in nodes_io_map:
                        # one input to Merge has been computed.
                        flagged_merge_nodes[trg][arg] = nodes_io_map[arg]

                if len(flagged_merge_nodes[trg]) == 2:
                    # both arguments to Merge have been computed.
                    if trg not in nodes_io_map:
                        nodes_io_map[trg] = {'input': None, 'output': None}

                        _output = self.modules_map[trg](
                            nodes_io_map[args[0]]['output'], 
                            nodes_io_map[args[1]]['output'])
                        
                        # Merge expands each input dim. into the max of that dim. between input tensors.
                        _input = torch.max(torch.stack([
                            nodes_io_map[args[0]]['output'], 
                            nodes_io_map[args[1]]['output']]), dim=0)

                        nodes_io_map[trg]['input'] = _input.values
                        nodes_io_map[trg]['output'] = _output

                # pass input through source.
                if src not in nodes_io_map:
                    nodes_io_map[src] = {'input': None, 'output': None}

                    if src == 0:
                        # first node in the graph.
                        _input = input_dummy
                        
                        # flag node being an input node of the network.
                        if src not in flagged_input_nodes:
                            flagged_input_nodes.append(src)

                    else:
                        # find node generating the input to be used.
                        inp_node = self._find_source_of_input_to(src)

                        if inp_node == -1:
                            #   `src` is receiving external (not from another layer) input. This will be the case when two
                            # parallel branches (two independent "input nodes" in the graph) merge at some point in the graph.
                            _input = input_dummy

                            # flag node being an input node of the network.
                            if src not in flagged_input_nodes:
                                flagged_input_nodes.append(src)
                        else:
                            if isinstance(self.modules_map[inp_node], sinabs.layers.merge.Merge):
                                # source of input is a `Merge` layer that might still need to have its I/O shapes computed.
                                self._handle_merge_source(inp_node, nodes_io_map)

                            print(f'accessing node {inp_node} cuz it is the input to node {src}....')

                            # record what the input shape for `src` should be.
                            _input = nodes_io_map[inp_node]['output']
                    
                    # forward input through the node.
                    _output = self.modules_map[src](_input)

                    # save node's input/output.
                    nodes_io_map[src] = {'input': _input, 'output': _output}

            else:
                # pass input through source.
                if src not in nodes_io_map:
                    nodes_io_map[src] = {'input': None, 'output': None}

                    if src == 0:
                        # first node in the graph.
                        _input = input_dummy

                        # flag node being an input node of the network.
                        if src not in flagged_input_nodes:
                            flagged_input_nodes.append(src)
                    else:
                        # find node generating the input to be used.
                        inp_node = self._find_source_of_input_to(src)

                        if inp_node == -1:
                            #   `src` is receiving external (not from another layer) input. This will be the case when two
                            # parallel branches (two independent "input nodes" in the graph) merge at some point in the graph.
                            _input = input_dummy
                            
                            # flag node being an input node of the network.
                            if src not in flagged_input_nodes:
                                flagged_input_nodes.append(src)
                        else:
                            if isinstance(self.modules_map[inp_node], sinabs.layers.merge.Merge):
                                # source of input is a `Merge` layer that might still need to have its I/O shapes computed.
                                self._handle_merge_source(inp_node, nodes_io_map)

                            # record what the input shape for `src` should be.
                            _input = nodes_io_map[inp_node]['output']
                    
                    # forward input through the node.
                    _output = self.modules_map[src](_input)

                    # save node's input/output.
                    nodes_io_map[src] = {'input': _input, 'output': _output}

                # pass input through target.
                if trg not in nodes_io_map:
                    nodes_io_map[trg] = {'input': None, 'output': None}

                    # find node generating the input to be used.
                    inp_node = self._find_source_of_input_to(trg)

                    if inp_node == -1:
                        #   `src` is receiving external (not from another layer) input. This will be the case when two
                        # parallel branches (two independent "input nodes" in the graph) merge at some point in the graph.
                        _input = input_dummy

                        # flag node being an input node of the network.
                        if trg not in flagged_input_nodes:
                            flagged_input_nodes.append(trg)
                    else:
                        if isinstance(self.modules_map[inp_node], sinabs.layers.merge.Merge):
                            # source of input is a `Merge` layer that might still need to have its I/O shapes computed.
                            self._handle_merge_source(inp_node, nodes_io_map)

                        # record what the input shape for `trg` should be.
                        _input = nodes_io_map[inp_node]['output']

                    # forward input through the node.
                    _output = self.modules_map[trg](_input)
                    
                    # save node's input/output.
                    nodes_io_map[trg] = {'input': _input, 'output': _output}

        # replace the I/O tensor information by its shape information.
        for node, io in nodes_io_map.items():
            nodes_io_map[node]['input'] = io['input'].shape
            nodes_io_map[node]['output'] = io['output'].shape

        return nodes_io_map, flagged_input_nodes
    
    def _handle_merge_source(self, merge_node_id: int, nodes_io_map: dict) -> None:
        """ This method finds the I/O shapes for node `merge_node_id` if they haven't been computed yet. When `self._find_source_of_input_to()` is 
        called the returned node might be a `Merge` layer for which the I/O shapes have yet to be computed.

        NOTE: In the current implemente both arguments to a `Merge` layer need to have the same output shapes.

        Parameters
        ----------
        - merge_node_id (int): the ID of the node representing a `Merge` layer.
        - nodes_io_map (dict): a dictionary mapping nodes to their I/O shapes.
        """

        if merge_node_id in nodes_io_map:
            # I/O shapes have been computed already.
            return None

        # finding nodes serving as argument to the `Merge` node...
        for edge in self._edges_list:

            if edge[1] == merge_node_id:
                # node `edge[0]` is one of the arguments for the `Merge` layer.
                if edge[0] in nodes_io_map:
                    # I/O shapes of one of the arguments for the `Merge` node has been computed.

                    # both arguments to `Merge` have the same I/O shape and merge outputs the same shape: updating I/O shape of `merge_node_id`.
                    nodes_io_map[merge_node_id] = {'input': nodes_io_map[edge[0]]['output'], 'output': nodes_io_map[edge[0]]['output']}

                    return None
                
        raise ValueError(f'Node {merge_node_id} is a \'Merge\' layer and I/O shape for none of its arguments have been computed yet.')

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

    def _find_merge_arguments(self, merge_node: int) -> list:
        """ A `Merge` layer receives two inputs. Return the two inputs to `merge_node` representing a `Merge` layer. """
        args = []
        for edge in self._edges_list:
            if edge[1] == merge_node:
                args.append(edge[0])
            if len(args) == 2:
                break
        return args