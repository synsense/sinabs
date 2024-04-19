import torch
import torch.nn as nn
import nirtorch
import copy
import sinabs
from typing import Tuple, Dict

class NIRtoDynapcnnNetworkGraph():
    def __init__(self, spiking_model, dummy_input) -> None:
        """ ."""

        nir_graph = nirtorch.extract_torch_graph(spiking_model, dummy_input, model_name=None).ignore_tensors()

        self.edges_list, self.name_2_indx_map = self.get_edges_from_nir(nir_graph)
        
        self.modules_map = self.get_named_modules(spiking_model)

        self.nodes_io_shapes = self.get_nodes_io_shapes(dummy_input)

    def get_edges_from_nir(self, nir_graph):
        """ ."""
        edges_list = []
        name_2_indx_map = {}
        idx_counter = 0

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
    
    def get_named_modules(self, model):
        """ ."""
        modules_map = {}

        if isinstance(model, nn.Sequential):                    # access modules via `.named_modules()`.
            for name, module in model.named_modules():
                if name != '':                                  # skip the module itself.
                    modules_map[self.name_2_indx_map[name]] = module

        elif isinstance(model, nn.Module):                      # access modules via `.named_children()`.
            for name, module in model.named_children():
                modules_map[self.name_2_indx_map[name]] = module

        else:
            raise ValueError('Either a nn.Sequential or a nn.Module is required.')

        return modules_map
    
    def get_nodes_io_shapes(self, input_dummy) -> Dict[int, Dict[str, torch.Size]]:
        """ ."""
        nodes_io_map = {}
        flagged_merge_nodes = {}

        for edge in self.edges_list:
            src = edge[0]
            trg = edge[1]

            if isinstance(self.modules_map[src], sinabs.layers.merge.Merge):
                # At this point the output of Merge has to have been calculated.
                
                # pass input through target.
                if trg not in nodes_io_map:
                    nodes_io_map[trg] = {'input': None, 'output': None}

                    inp_node = self.find_input_to_node(trg)                     # find node generating the input to be used.
                    _input = nodes_io_map[inp_node]['output']

                    _output = self.modules_map[trg](_input)                     # forward input through the node.

                    nodes_io_map[trg] = {'input': _input, 'output': _output}  # save node's input/output.

            elif isinstance(self.modules_map[trg], sinabs.layers.merge.Merge):
                # Merge requires two inputs: need to check if both of its inputs have been calculated.
                if trg not in flagged_merge_nodes:
                    flagged_merge_nodes[trg] = {}
                
                args = self.find_merge_arguments(trg)

                for arg in args:
                    if arg in nodes_io_map:                          # one input to Merge has been computed.
                        flagged_merge_nodes[trg][arg] = nodes_io_map[arg]

                if len(flagged_merge_nodes[trg]) == 2:                 # both arguments to Merge have been computed.
                    if trg not in nodes_io_map:
                        nodes_io_map[trg] = {'input': None, 'output': None}

                        _output = self.modules_map[trg](
                            nodes_io_map[args[0]]['output'], 
                            nodes_io_map[args[1]]['output'])
                        
                        _input = torch.max(torch.stack([                # Merge expands each input dim. into the max of that dim. between input tensors.
                            nodes_io_map[args[0]]['output'], 
                            nodes_io_map[args[1]]['output']]), dim=0)

                        nodes_io_map[trg]['input'] = _input.values
                        nodes_io_map[trg]['output'] = _output

                # pass input through source.
                if src not in nodes_io_map:
                    nodes_io_map[src] = {'input': None, 'output': None}

                    if src == 0:
                        _input = input_dummy                                    # first node in the graph.
                    else:
                        inp_node = self.find_input_to_node(src)                 # find node generating the input to be used.
                        _input = nodes_io_map[inp_node]['output']
                    
                    _output = self.modules_map[src](_input)                     # forward input through the node.

                    nodes_io_map[src] = {'input': _input, 'output': _output}  # save node's input/output.

            else:

                # pass input through source.
                if src not in nodes_io_map:
                    nodes_io_map[src] = {'input': None, 'output': None}

                    if src == 0:
                        _input = input_dummy                                    # first node in the graph.
                    else:
                        inp_node = self.find_input_to_node(src)                 # find node generating the input to be used.
                        _input = nodes_io_map[inp_node]['output']
                    
                    _output = self.modules_map[src](_input)                     # forward input through the node.

                    nodes_io_map[src] = {'input': _input, 'output': _output}  # save node's input/output.

                # pass input through target.
                if trg not in nodes_io_map:
                    nodes_io_map[trg] = {'input': None, 'output': None}

                    inp_node = self.find_input_to_node(trg)                     # find node generating the input to be used.
                    _input = nodes_io_map[inp_node]['output']

                    _output = self.modules_map[trg](_input)                     # forward input through the node.

                    nodes_io_map[trg] = {'input': _input, 'output': _output}  # save node's input/output.

        for node, io in nodes_io_map.items():
            nodes_io_map[node]['input'] = io['input'].shape
            nodes_io_map[node]['output'] = io['output'].shape

        return nodes_io_map

    def find_input_to_node(self, node):
        """ ."""
        for edge in self.edges_list:
            if edge[1] == node:
                return edge[0]
        return -1
    
    def find_node_variable_name(self, node):
        """ ."""
        for key, val in self.name_2_indx_map.items():
            if val == node:
                return key
        return None

    def find_merge_arguments(self, merge_node):
        """ ."""
        args = []
        for edge in self.edges_list:
            if edge[1] == merge_node:
                args.append(edge[0])
            if len(args) == 2:
                break
        return args
    
    def remove_ignored_nodes(self, default_ignored_nodes):
        """ Recreates the edges list based on layers that 'DynapcnnNetwork' will ignore. This
        is done by setting the source (target) node of an edge where the source (target) node
        will be dropped as the node that originally targeted this node to be dropped.
        """
        edges = copy.deepcopy(self.edges_list)
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
    
    def get_node_io_shapes(self, node) -> Tuple[torch.Size, torch.Size]:
        """ ."""
        return self.nodes_io_shapes[node]['input'], self.nodes_io_shapes[node]['output']
