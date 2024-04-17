import torch
import torch.nn as nn
import nirtorch
import copy

class NIRtoDynapcnnNetworkGraph():
    def __init__(self, analog_model, dummy_input) -> None:
        """ .

        TODO
            [ ] test it with nn.Sequential and a version defined as class inheriting from nn.Module.
        """

        nir_graph = nirtorch.extract_torch_graph(analog_model, dummy_input, model_name=None).ignore_tensors()

        self.edges_list, self.name_2_indx_map = self.get_edges_from_nir(nir_graph, analog_model)
        
        self.modules_map = self.get_named_modules(analog_model)

    def get_edges_from_nir(self, nir_graph, analog_model):
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
    
    def get_named_modules(self, analog_model):
        """ ."""
        modules_map = {}

        if isinstance(analog_model, nn.Sequential):             # access modules via `.named_modules()`.
            for name, module in analog_model.named_modules():
                if name != '':                                  # skip the module itself.
                    modules_map[self.name_2_indx_map[name]] = module

        elif isinstance(analog_model, nn.Module):               # access modules via `.named_children()`.
            for name, module in analog_model.named_children():
                modules_map[self.name_2_indx_map[name]] = module

        else:
            # TODO raise error
            pass

        return modules_map
    
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