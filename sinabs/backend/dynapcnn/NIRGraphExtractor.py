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
        
        # recovers the associated `nn.Module` (layer) of each node.
        self.modules_map = self._get_named_modules(spiking_model)

        # retrieves what the I/O shape for each node's module is.
        self._nodes_io_shapes = self._get_nodes_io_shapes(dummy_input)

    ### Publich Methods ###

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
    
    def _get_nodes_io_shapes(self, input_dummy: torch.tensor) -> Dict[int, Dict[str, torch.Size]]:
        """ Loops through the graph represented in `self._edges_list` and propagates the inputs through the nodes, starting from
        `node 0` fed `input_dummy`.
        """
        nodes_io_map = {}
        flagged_merge_nodes = {}

        # propagate inputs through the nodes.
        for edge in self._edges_list:
            src = edge[0]
            trg = edge[1]

            if isinstance(self.modules_map[src], sinabs.layers.merge.Merge):
                # At this point the output of Merge has to have been calculated.
                
                # pass input through target.
                if trg not in nodes_io_map:
                    nodes_io_map[trg] = {'input': None, 'output': None}

                    # find node generating the input to be used.
                    inp_node = self._find_input_to_node(trg)
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

                    else:
                        # find node generating the input to be used.
                        inp_node = self._find_input_to_node(src)
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
                    else:
                        # find node generating the input to be used.
                        inp_node = self._find_input_to_node(src)
                        _input = nodes_io_map[inp_node]['output']
                    
                    # forward input through the node.
                    _output = self.modules_map[src](_input)

                    # save node's input/output.
                    nodes_io_map[src] = {'input': _input, 'output': _output}

                # pass input through target.
                if trg not in nodes_io_map:
                    nodes_io_map[trg] = {'input': None, 'output': None}

                    # find node generating the input to be used.
                    inp_node = self._find_input_to_node(trg)
                    _input = nodes_io_map[inp_node]['output']

                    # forward input through the node.
                    _output = self.modules_map[trg](_input)
                    
                    # save node's input/output.
                    nodes_io_map[trg] = {'input': _input, 'output': _output}

        # replace the I/O tensor information by its shape information.
        for node, io in nodes_io_map.items():
            nodes_io_map[node]['input'] = io['input'].shape
            nodes_io_map[node]['output'] = io['output'].shape

        return nodes_io_map

    def _find_input_to_node(self, node: int) -> int:
        """ Finds the first edge `(X, node)` returns `X`. """
        for edge in self._edges_list:
            if edge[1] == node:
                return edge[0]
        raise ValueError(f'Node {node} is not the target node of any edge in the graph.')

    def _find_merge_arguments(self, merge_node: int) -> list:
        """ A `Merge` layer receives two inputs. Return the two inputs to `merge_node` representing a `Merge` layer. """
        args = []
        for edge in self._edges_list:
            if edge[1] == merge_node:
                args.append(edge[0])
            if len(args) == 2:
                break
        return args