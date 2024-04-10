from typing import Tuple
import torch.nn as nn
import sinabs.layers as sl

# @TODO this constraints are ideally device-dependent.
VALID_SINABS_EDGES = {
    0: (nn.Conv2d, sl.iaf.IAFSqueeze),
    1: (sl.iaf.IAFSqueeze, nn.AvgPool2d),
    2: (sl.iaf.IAFSqueeze, nn.Conv2d),
    3: (sl.iaf.IAFSqueeze, nn.Linear),
    4: (nn.AvgPool2d, nn.Conv2d),
    5: (nn.AvgPool2d, nn.Linear),
    6: (nn.Linear, sl.iaf.IAFSqueeze),
}

VALID_SINABS_NODE_FAN_IN = []
VALID_SINABS_NODE_FAN_OUT = []

def process_edge(layers: list, edge: Tuple[int, int], mapper: dict):
    """ ."""
    edge_layers = (type(layers[edge[0]]), type(layers[edge[1]]))
    edge_type = is_valid_edge(edge_layers)

    if edge_type:
        # incorporate modules within the edge to one DynapcnnLayer.
        update_dynapcnnlayer_mapper(edge_type, edge, mapper, layers)
    else:
        raise TypeError(f'Invalid graph edge: {edge_layers}')
    
def is_valid_edge(edge):
    """. """
    for key, edge_type in VALID_SINABS_EDGES.items():
        if edge == edge_type:
            return key
    return None
    
def update_dynapcnnlayer_mapper(edge_type: int, edge: Tuple[int, int], mapper: dict, layers: list):
    """ ."""
    if edge_type == 0:            # (conv, iaf): has to be a new DynapcnnLayer -> @TODO not necessarily! See 'edge_type == 2'.
        new_key = 0
        for indx, layers_set in mapper.items(): # @TODO have to check if node for conv exists on 'mapper' (see 'edge_type == 2').
            new_key += 1
        mapper[new_key] = {edge[0]: layers[edge[0]], edge[1]: layers[edge[1]]}
    elif edge_type == 1:          # (iaf, pool): pool has to be part of a previously initialized DynapcnnLayer.
        matched = False
        for indx, layers_set in mapper.items():
            if layers[edge[0]] == layers_set[edge[1]]:
                mapper[indx][edge[1]] = layers[edge[1]]
                matched = True
                break
        if not matched:
            raise TypeError(f'Edge {edge} can not be matched to already mapped layers.')
    elif edge_type == 2:          # (iaf, conv): must be an edge between an existing DynapcnnLayers and a new one.
        matched = False
        for indx, layers_set in mapper.items():
            for node_indx, mod in layers_set.items():
                if node_indx == edge[0]:
                    mapper[indx+1] = {edge[1]: layers[edge[1]]}
                    matched = True
                    break
            if matched:
                break
        if not matched:
            raise TypeError(f'Edge {edge} can not be matched to already mapped layers.')
    elif edge_type == 3:          # (, ): ...
        ...
    elif edge_type == 4:          # (, ): ...
        ...
    elif edge_type == 5:          # (, ): ...
        ...
    elif edge_type == 6:          # (, ): ...
        ...
    else:
        raise TypeError(f'Invalid graph edge type: {edge_type}')
