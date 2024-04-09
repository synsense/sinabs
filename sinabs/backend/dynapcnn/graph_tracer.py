import torch
import torch.nn as nn
import re, copy
import numpy as np
import networkx as nx
from typing import Union

import matplotlib.pyplot as plt

class GraphTracer():
    def __init__(self, model: Union[nn.Sequential, nn.Module], dummy_input: np.array) -> None:
        """ ."""

        trace = torch.jit.trace(model, dummy_input)
        _ = trace(dummy_input)
        __ = copy.deepcopy(trace)

        self.graph = __.graph

        self.modules_map, self.name_2_indx_map  = self.get_named_modules(model)
        self.forward_edges                      = self.get_foward_edges()
        self.ATens                              = self.get_ATen_operations()
        self.edges_list                         = self.get_graph_edges()

    def from_name_2_indx(self, name):
        if name in self.name_2_indx_map:
            return self.name_2_indx_map[name]
        else:
            last_indx = None
            for _name, indx in self.name_2_indx_map.items():
                last_indx = indx
            self.name_2_indx_map[name] = last_indx+1
            return self.name_2_indx_map[name]

    def get_named_modules(self, module: nn.Module):
        """ ."""
        modules_map = {}
        name_2_indx_map = {}
        indx = 0
        for name, mod in module.named_modules():
            if name:
                modules_map[indx] = mod
                name_2_indx_map[name] = indx
                indx += 1
        return modules_map, name_2_indx_map
    
    def get_foward_edges(self):
        """ ."""
        forward_edges = {}
        for node in self.graph.nodes():
            node = str(node)
            regex = re.compile(r'%(.*?) :.*prim::CallMethod\[name="forward"\]\(%(.*?), %(.*?)\)')
            match = regex.search(node)
            if match:
                source = match.group(3).replace('_', '')
                target = match.group(2).replace('_', '')
                result = match.group(1).replace('_', '')
                forward_edges[self.from_name_2_indx(result)] = (self.from_name_2_indx(source), self.from_name_2_indx(target))
                
        return forward_edges

    def get_graph_edges(self):
        """ ."""
        edges = []
        last_result = None

        for result_node, forward_edge in self.forward_edges.items():
            src = forward_edge[0]
            trg = forward_edge[1]

            if not last_result:
                last_result = result_node
                edges.append(('input', trg))
            elif src == last_result:
                edges.append((edges[-1][1], trg))
                last_result = result_node
            else:
                scr1, scr2 = self.get_ATen_operands(src)
                edges.append((scr1, trg))
                edges.append((scr2, trg))
                last_result = result_node
    
        edges.append((edges[-1][1], 'output'))

        return edges[1:-1]
    
    def get_ATen_operands(self, node):
        """ ."""
        if node in self.ATens:
            src1 = self.ATens[node]['args'][1]
            src2 = self.ATens[node]['args'][0]
            return self.forward_edges[src1][1], self.forward_edges[src2][1]
        else:
            # throw error
            return None, None
        
    def get_ATen_operations(self):
        """ ATen is PyTorch's tensor library backend, which provides a set of operations that operate on 
        tensors directly. These include arithmetic operations (add, mul, etc.), mathematical 
        functions (sin, cos, etc.), and tensor manipulation operations (view, reshape, etc.)."""
        ATens = {}
        for node in self.graph.nodes():
            node = str(node)
            regex = re.compile(r'%(.*?) :.*aten::(.*?)\(%(.*?), %(.*?), %(.*?)\)')

            match = regex.search(node)

            if match:
                result_node = match.group(1)
                operation = match.group(2)
                operator1 = self.from_name_2_indx(match.group(3))
                operator2 = self.from_name_2_indx(match.group(4))
                const_operator = match.group(5)
                ATens[result_node] = {'op': operation, 'args': (operator1, operator2, const_operator)}
        return ATens
    
    def remove_ignored_nodes(self, default_ignored_nodes):
        """ Recreates the edges list based on layers that 'DynapcnnNetwork' will ignored. This
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

        return remapped_edges, parsed_edges
    
    def plot_graph(self):
        """ ."""
        G = nx.DiGraph(self.edges_list)
        layout = nx.spring_layout(G)
        nx.draw(G, pos = layout, with_labels=True, node_size=800)
        plt.title('GraphTracer (new)')
        plt.show()
