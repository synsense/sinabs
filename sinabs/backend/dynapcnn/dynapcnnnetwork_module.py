# functionality : ...
# author        : Willian Soares Girao
# contact       : williansoaresgirao@gmail.com

import torch.nn as nn
from typing import List, Tuple, Dict

class DynapcnnNetworkModule(nn.Module):
    """ ."""
    def __init__(self, dcnnl_edges: List[Tuple[int, int]], dynapcnn_layers: Dict) -> nn.Module:
        super().__init__()

        self.model_forward = self._build_module_forward_from_graph(dcnnl_edges, dynapcnn_layers)

    def _build_module_forward_from_graph(self, dcnnl_edges, dynapcnn_layers):
        """
        ...

        TODO the `Merge` layer has to be recreated here if a node appears as the targert in more than one edge.
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

        print('original edges: ')
        for edge in dcnnl_edges:
            print(edge)
        
        print('\nforward edges: ')
        for edge in new_edges_set:
            print(edge)

        return forward_map
