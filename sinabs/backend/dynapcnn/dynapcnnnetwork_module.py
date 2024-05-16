# functionality : ...
# author        : Willian Soares Girao
# contact       : williansoaresgirao@gmail.com

import torch.nn as nn
from sinabs.layers import Merge
from typing import List, Tuple, Dict, Union
import copy
import sinabs
import sinabs.layers as sl

class DynapcnnNetworkModule():
    """
    Uses the set of `DynapcnnLayer` instances and how they address each other to define what the `forward` method of the model should do.

    Parameters
    ----------
        dcnnl_edges (list): tuples representing the output->input mapping between `DynapcnnLayer` instances
            that have been used as configuration for each core `CNNLayerConifg`.
        dynapcnn_layers (dict): the `DynapcnnLayer` instances along with their supporting metadata (e.g. assigned core,
            destination layers, etc.).
    """

    def __init__(self, dcnnl_edges: List[Tuple[int, int]], dynapcnn_layers: Dict):

        self.dcnnl_edges = dcnnl_edges

        self.forward_map, self.merge_points = self._build_module_forward_from_graph(dcnnl_edges, dynapcnn_layers)
    
    def _spot_merging_points(self, dcnnl_edges: list) -> dict:
        """ . """

        nodes_with_merge_input = {}

        for edge in dcnnl_edges:
            trg_node = edge[1]
            fan_in = 0
            src_nodes = []

            for edge_inner in dcnnl_edges:
                if edge_inner[1] == trg_node:
                    fan_in += 1
                    src_nodes.append(edge_inner[0])

            if fan_in == 2 and trg_node not in nodes_with_merge_input:
                nodes_with_merge_input[trg_node] = {'sources': tuple(src_nodes), 'merge': sl.Merge()}
            
            if fan_in > 2:
                raise ValueError(f'Node {trg_node} is the has fan-in of {fan_in}: only fan-in of 2 is currently handled.')
            
        return nodes_with_merge_input
    
    def _build_module_forward_from_graph(self, dcnnl_edges: list, dynapcnn_layers: dict) -> Union[dict, dict]:
        """ ."""

        # mapper to flag nodes that need input from a `Merge` layer.
        merge_points = self._spot_merging_points(dcnnl_edges)

        # this dict. will be used to call the `forward` methods of each `DynapcnnLayer`.
        forward_map = {}

        for edge in dcnnl_edges:
            src_dcnnl = edge[0]     # source layer
            trg_dcnnl = edge[1]     # target layer

            if src_dcnnl not in forward_map:
                forward_map[src_dcnnl] = copy.deepcopy(dynapcnn_layers[src_dcnnl]['layer'])
            
            if trg_dcnnl not in forward_map:
                forward_map[trg_dcnnl] = copy.deepcopy(dynapcnn_layers[trg_dcnnl]['layer'])

        return forward_map, merge_points