# functionality : ...
# author        : Willian Soares Girao
# contact       : williansoaresgirao@gmail.com

import torch.nn as nn
from sinabs.layers import Merge
from typing import List, Tuple, Dict, Union
import copy
import sinabs
import sinabs.layers as sl

class DynapcnnNetworkModule(nn.Module):
    """
        Uses the set of `DynapcnnLayer` instances that have been configured to the chip and how they address each other
    to define what the `forward` method of the model should do.

    Parameters
    ----------
        dcnnl_edges (list): tuples representing the output->input mapping between `DynapcnnLayer` instances
            that have been used as configuration for each core `CNNLayerConifg`.
        dynapcnn_layers (dict): the `DynapcnnLayer` instances along with their supporting metadata (e.g. assigned core,
            destination layers, etc.).
    """

    def __init__(self, dcnnl_edges: List[Tuple[int, int]], dynapcnn_layers: Dict) -> nn.Module:
        super().__init__()

        self._forward_edges, self._forward_map = self._build_module_forward_from_graph(dcnnl_edges, dynapcnn_layers)

    def _build_module_forward_from_graph(self, dcnnl_edges: list, dynapcnn_layers: dict) -> Union[list, dict]:
        """
        TODO use copy.deepcopy for create the `forward_map`.
        """
        forward_map = {}
        new_edges_set = []
        divergent_nodes = []
        
        for edge in dcnnl_edges:
            source_dcnnl = edge[0]
            target_dcnnl = edge[1]

            new_edge_2_append = []

            # processing the source `DynapcnnLayer`.

            if source_dcnnl not in forward_map:
                forward_map[source_dcnnl] = dynapcnn_layers[source_dcnnl]['layer']

                if len(forward_map[source_dcnnl].pool_layer) > 1:
                    # this `DynapcnnLayer` is a divergent point in the graph.
                    divergent_nodes.append(source_dcnnl)
                    for i in range(len(forward_map[source_dcnnl].pool_layer)):
                        
                        # create edge representing forward through the i-th pooling layer.
                        pool_name = f'{source_dcnnl}_pool{i}'
                        new_edges_set.append((source_dcnnl, pool_name))

                        # create forward 'node' for the i-th pooling layer.
                        if pool_name not in forward_map:
                            forward_map[pool_name] = forward_map[source_dcnnl].pool_layer[i]

                        # create edge from i-th pooling to its target `DynapcnnLayer`.
                        new_edge_2_append.append((pool_name, dynapcnn_layers[source_dcnnl]['destinations'][i]))

            # processing the target `DynapcnnLayer`.

            if target_dcnnl not in forward_map:
                forward_map[target_dcnnl] = dynapcnn_layers[target_dcnnl]['layer']

                if len(forward_map[target_dcnnl].pool_layer) > 1:
                    # this `DynapcnnLayer` is a divergent point in the graph.
                    divergent_nodes.append(target_dcnnl)
                    for i in range(len(forward_map[target_dcnnl].pool_layer)):
                        
                        # create edge representing forward through the i-th pooling layer.
                        pool_name = f'{target_dcnnl}_pool{i}'
                        new_edges_set.append((target_dcnnl, pool_name))

                        # create forward 'node' for the i-th pooling layer.
                        if pool_name not in forward_map:
                            forward_map[pool_name] = forward_map[target_dcnnl].pool_layer[i]

                        # create edge from i-th pooling to its target `DynapcnnLayer`.
                        new_edge_2_append.append((pool_name, dynapcnn_layers[target_dcnnl]['destinations'][i]))

            if source_dcnnl not in divergent_nodes and target_dcnnl not in divergent_nodes:
                # save original edge.
                new_edges_set.append(edge)

            if len(new_edge_2_append) != 0:
                new_edges_set.extend(new_edge_2_append)

        forward_edges = self._find_merging_nodes(new_edges_set, forward_map)

        return forward_edges, forward_map
    
    def _find_merging_nodes(self, edges_list: list, forward_map: dict) -> list:
        """ Loops through the edges and see if a node appeards in more than one edge. If so, this is a node
        that requires a `Merge` layer. For instance, edges `(A, X)` and `(B, X)` will be replace by two new
        edges `((A, B), Merge_X)` and `(Merge_X, X)`, where `A` and `B` are the inputs to a `Merge` feeding into `X`.
        """
        merge_mapping = {}

        for edge in edges_list:
            src = edge[0]
            trg = edge[1]

            if trg in merge_mapping:
                # node needs to receive input from a `Merge` layer.
                merge_arguments = (
                    merge_mapping[trg]['src'],         # merge_arguments[0] = source (from 1st edge containing `trg`).
                    src)                               # merge_arguments[1] = `src` (the source of the 2nd edge containing `trg`).

                merge_mapping[trg] = {'src': merge_arguments} 

            else:
                merge_mapping[trg] = {'src': src}

        final_edges = []
        merge_idx = 0

        # create edges `((A, B), Merge_X)` and `(Merge_X, X)`.
        for trg, src in merge_mapping.items():
            _ = src['src']

            if isinstance(_, tuple):
                # `trg` receives from a `Merge` layer.
                merge_node = f'merge_{merge_idx}'
                forward_map[merge_node] = Merge()
                
                new_edge = (_, merge_node)
                final_edges.append(new_edge)

                new_edge = (merge_node, trg)
                final_edges.append(new_edge)

                merge_idx += 1

            else:
                final_edges.append((_, trg))
        
        return final_edges
    
    def parameters(self) -> list:
        """ ."""
        parameters = []

        for module in self._forward_map.values():
            if isinstance(module, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                parameters.extend(module.conv_layer.parameters())

        return parameters
    
    def init_weights(self):
        """ ."""
        for node, module in self._forward_map.items():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight.data)

    def detach_neuron_states(self) -> None:
        """ Detach the neuron states and activations from current computation graph (necessary). """

        for module in self._forward_map.values():
            if isinstance(module, sinabs.backend.dynapcnn.dynapcnn_layer_new.DynapcnnLayer):
                if isinstance(module.spk_layer, sl.StatefulLayer):
                    for name, buffer in module.spk_layer.named_buffers():
                        buffer.detach_()
    
    def forward(self, x):
        """ The torch forward uses `self._forward_edges` to feed data throguh the 
        layers in `self._forward_map`.
        """

        layers_outputs = {}

        # input node has to be `0`.
        layers_outputs[0] = self._forward_map[0](x)

        for edge in self._forward_edges:
            src = edge[0]
            trg = edge[1]

            # gets the input to the target node (must have been computed already).
            if isinstance(src, tuple):
                # `trg` is a Merge layer.
                arg1 = layers_outputs[src[0]]
                arg2 = layers_outputs[src[1]]

                layers_outputs[trg] = self._forward_map[trg](arg1, arg2)
                
            else:
                x = layers_outputs[src]
                layers_outputs[trg] = self._forward_map[trg](x)

        return layers_outputs[trg]