"""
functionality : functions implementing the pre-processing of edges into blocks of nodes (modules) for future
                creation of DynapcnnLayer objects.
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

from typing import Tuple, List, Dict
import sinabs.layers
import torch.nn as nn
from .sinabs_edges_utils import *
import sinabs, copy

def process_edge(layers: Dict[int, nn.Module], edge: Tuple[int, int], mapper: dict) -> None:
    """ Read in an edge describing the connection between two layers (nodes in the computational graph). If `edge`
    is a valid connection between two layers, update `mapper` to incorporate these layers into a new or existing dictonary
    containing the modules comprising a future `DynacnnLayer` object.

    After of call of this function `mapper` is updated to incorporate a set of nodes into the data required to create a
    `DynapcnnLayer` instance. For example, after processing the 1st edge `(0, 1)`, an entry `0` for a future `DynapcnnLayer` is
    created and its set of nodes will include node `0` and node `1`:

    mapper[0] = {
        0: {'layer': Conv2d, 'input_shape': None, 'output_shape': None}, 
        1: {'layer': IAFSqueeze, 'input_shape': None, 'output_shape': None},
        ...
    }

    Parameters
    ----------
        layers (dict): a dictionary containing the nodes of the graph as `key` and their associated module as `value`.
        edge (tuple): tuple representing the connection between two nodes in computational graph of 'DynapcnnNetworkGraph.snn.spiking_model'.
        mapper (dict): dictionary where each 'key' is the index of a future 'DynapcnnLayer' and 'value' is a dictionary ('key': node, 'value': module).
    """
    edge_type = is_valid_edge(edge, layers, VALID_SINABS_EDGES)

    if isinstance(edge_type, int):                                  # incorporate modules within the edge to a dict representing a future DynapcnnLayer.
        update_dynapcnnlayer_mapper(edge_type, edge, mapper, layers)
    else:
        raise InvalidEdge(edge, type(layers[edge[0]]), type(layers[edge[1]]))
    
def is_valid_edge(edge: Tuple[int, int], layers: Dict[int, nn.Module], valid_edges_map: dict) -> int:
    """ Checks if the modules each node in 'edge' represent are a valid connection between a sinabs network to be 
    loaded on Speck.

    Parameters
    ----------
        valid_edges_map: dictionary where each 'key' is the type (index) of a pre-defined valid edge.

    Returns
    ----------
        edge_type: the edge type specified in 'valid_edges_map' ('None' if edge is not valid).
    """
    edge_layers = (layers[edge[0]], layers[edge[1]])

    for edge_type, sinabs_edge in valid_edges_map.items():
        if (type(edge_layers[0]) == sinabs_edge[0]) and (type(edge_layers[1]) == sinabs_edge[1]):
            return edge_type

    return None
    
def update_dynapcnnlayer_mapper(edge_type: int, edge: Tuple[int, int], mapper: dict, layers: Dict[int, nn.Module]) -> None:
    """ Parses the nodes within an edge and incorporate them either into a **new** or an **already existing** DynapcnnLayer represented 
    in 'mapper'.
    """

    if edge_type in [0, 6]:
        init_xor_complete_new_dynapcnnlayer_blk(mapper, edge, layers)

    elif edge_type in [1, 7]:
        add_pool_to_dynapcnnlayer_blk(mapper, edge, layers)
        
    elif edge_type in [2, 3, 4, 5, 8, 9]:
        connect_dynapcnnlayer_blks(mapper, edge, layers)

    else:
        raise InvalidEdgeType(edge, edge_type)
    
def init_xor_complete_new_dynapcnnlayer_blk(mapper: dict, edge: Tuple[int, int], layers: Dict[int, nn.Module]) -> None:
    """ Incorporates nodes from either a `(conv, neuron)` or a `(linear, neuron)` edge. These are either initiating a new `dict` mapping
    into a future `DynapcnnLayer` or completing a `conv->neuron` sequence (in the case the node for `conv` as already been incorporated 
    somewhere in `mapper`). Obs.: `nn.Linear` layers are converted into `nn.Conv2d` by `DynapcnnLayer`.
    """
    matched = False
    dynapcnnlayer_indx = 0

    for indx, dynapcnnlayer in mapper.items():                      # see if 'edge[0]' exists in a DynapcnnLayer block.
        for node, _ in dynapcnnlayer.items():
            if node == edge[0]:
                dynapcnnlayer_indx = indx
                matched = True
                break
        if matched:                                                 # 'edge[0]' found: 'edge[1]' belongs to its DynapcnnLayer block.
            mapper[dynapcnnlayer_indx][edge[1]] = {'layer': layers[edge[1]], 'input_shape': None, 'output_shape': None}
            break
        
    if not matched:                                                 # 'edge[0]' not found: start new DynapcnnLayer block.
        dynapcnnlayer_indx = 0
        for indx, _ in mapper.items():
            dynapcnnlayer_indx += 1
        mapper[dynapcnnlayer_indx] = {
            edge[0]: {'layer': layers[edge[0]], 'input_shape': None, 'output_shape': None}, 
            edge[1]: {'layer': layers[edge[1]], 'input_shape': None, 'output_shape': None}
            }

def connect_dynapcnnlayer_blks(mapper: dict, edge: Tuple[int, int], layers: Dict[int, nn.Module]) -> None:
    """ Incorporates nodes from either a `(neuron, conv)/(neuron, lin)` or `(pool, conv)/(pool, lin)` edge. These represent connections between an existing 
    `dict` in `mapper` that will be mapped into a `DynapcnnLayer` and a new one yet to be represented in `mapper`. Obs.: `nn.Linear` layers are converted 
    into `nn.Conv2d` by `DynapcnnLayer`.
    """
    if not is_initialized_node(edge[1], mapper):
        dynapcnnlayer_indx = 0
        matched = False
        for indx, dynapcnnlayer in mapper.items():
            for node, _ in dynapcnnlayer.items():
                if node == edge[0]:                                 # 'edge[0]' is ending DynapcnnLayer block 'indx'.
                    dynapcnnlayer_indx = indx+1
                    matched = True
                    break
            if matched:
                break
        if matched:
            while (dynapcnnlayer_indx in mapper):
                dynapcnnlayer_indx += 1
            mapper[dynapcnnlayer_indx] = {                          # 'edge[1]' starts new DynapcnnLayer block.
                edge[1]: {'layer': layers[edge[1]], 'input_shape': None, 'output_shape': None}}
        else:
            raise UnmatchedNode(edge, node)
    
def add_pool_to_dynapcnnlayer_blk(mapper: dict, edge: Tuple[int, int], layers: Dict[int, nn.Module]) -> None:
    """ Incorporating a `(neuron, pool)` edge. Node `pool` has to be part of an already existing `dict` mapping into a `DynapcnnLaye` in `mapper`.
    """
    matched = False
    for indx, dynapcnnlayer in mapper.items():
        for node, _ in dynapcnnlayer.items():
            if node == edge[0]:
                dynapcnnlayer[edge[1]] = {                          # 'edge[0]' is a neuron layer inputing into pooling layer 'edge[1]'.
                    'layer': layers[edge[1]], 'input_shape': None, 'output_shape': None}
                matched = True
                break
        if matched:
            break
    if not matched:
        raise UnmatchedNode(edge, node)
    
def is_initialized_node(node: int, mapper: dict) -> bool:
    """ Finds if 'node' existis within 'mapper'. """
    for _, dynapcnnlayer in mapper.items():
        for _node, __ in dynapcnnlayer.items():
            if _node == node:
                return True
    return False

def get_dynapcnnlayers_destinations(layers: Dict[int, nn.Module], edges: List[Tuple[int, int]], mapper: dict) -> dict:
    """ Loops over the edges list describing the computational graph. It will access each node in the graph and find to which
    DynapcnnLayer they belong to. If source and target belong to different DynapcnnLayers (described as a dictionary in 'mapper')
    the destination of the 'DynapcnnLayer.source' is set to be 'DynapcnnLayer.target'.

    After one call of this function an attribute `destination` is added to an entry in `mapper` to save the indexes (a different `key`
    in `mapper`) of `DynapcnnLayer`s targeted by another `DynapcnnLayer`. For example, if in an edge `(1, 4)` the node `1` belongs to
    `mapper[0]` and node `4` belongs to `mapper[2]`, the former is updated to tager the latter, like the following:

    mapper[0] = {
        0: {'layer': Conv2d, ...}, 
        1: {'layer': IAFSqueeze, ...},  # node `1` in edge `(1, 4)` belongs to `mapper[0]`...
        ...
        'destinations': [2],            # ... so DynacnnLayer built from `mapper[2]` is destination of DynapcnnLayer built from `mapper[0]`.
        ...
    }

    Parameters
    ----------
        layers (dict): contains the nodes of the graph as `key` and their associated module as `value`.
        edges (list): tuples representing the connection between nodes in computational graph spiking network.
        mapper (dict): each 'key' is the index of a future `DynapcnnLayer` and `value` the data necessary to instantiate it.

    Returns
    ----------
        dynapcnnlayers_destinations_map: dictionary where each 'key' is the index of a future 'DynapcnnLayer' and 'value' is its list of destinations (DynapcnnLayers).
    """
    dynapcnnlayers_destinations_map = {}
    used_layer_edges = []

    for edge in edges:
        source_layer = get_dynapcnnlayer_index(edge[0], mapper)
        destination_layer = get_dynapcnnlayer_index(edge[1], mapper)

        if source_layer not in dynapcnnlayers_destinations_map:
            dynapcnnlayers_destinations_map[source_layer] = []

        if source_layer != destination_layer and is_valid_dynapcnnlayer_pairing(layers, edge, VALID_DYNAPCNNLAYER_EDGES):
            # valid connection between modules in two different DynapcnnLayer.

            if len(dynapcnnlayers_destinations_map[source_layer]) > 2:
                # DynapcnnLayers can not have more than two destinations.
                raise MaxDestinationsReached(source_layer)
            else:
                if (destination_layer, source_layer) not in used_layer_edges and destination_layer not in dynapcnnlayers_destinations_map[source_layer]:
                    # edge does not create a loop between layers.
                    dynapcnnlayers_destinations_map[source_layer].append(destination_layer)
                    used_layer_edges.append((source_layer, destination_layer))
                else:
                    raise InvalidLayerLoop(source_layer, destination_layer)
                
    for dcnnl_idx, destinations in dynapcnnlayers_destinations_map.items():
        # TODO document the 'rescale_factor' better.
        mapper[dcnnl_idx]['destinations'] = destinations
        mapper[dcnnl_idx]['conv_rescale_factor'] = []
                
def get_dynapcnnlayer_index(node: int, mapper: dict) -> int:
    """ Returns the DynapcnnLayer index to which 'node' belongs to. """
    for indx, dynapcnnlayer in mapper.items():
        if node in dynapcnnlayer:
            return indx
    raise UnknownNode(node)

def is_valid_dynapcnnlayer_pairing(layers: Dict[int, nn.Module], edge: Tuple[int, int], valid_dynapcnnlayer_edges: List[Tuple[nn.Module, nn.Module]]) -> bool:
    """ Checks if the module in 'DynapcnnLayer.source' is targetting a valid module in 'DynapcnnLayer.target'. """
    if (type(layers[edge[0]]), type(layers[edge[1]])) in valid_dynapcnnlayer_edges:
        return True
    else:
        raise InvalidLayerDestination(type(layers[edge[0]]), type(layers[edge[1]]))
    
def merge_handler(sinabs_edges: List[Tuple[int, int]], sinabs_modules_map: Dict[int, nn.Module]) -> List[Tuple[int, int]]:
    """ Handles connections between nodes made via a `sinabs.layers.Merge` layer. If `X` is a merge layer then edges `(X, C)` are removed
    from the edges list since they don't affect the creationg of `DynapcnnLayer`s. Edges `(Y, X)` are turned into a edge `(Y, C)` pointing
    directly to the node receiving the merged inputs such that the `DynapcnnLayer` containing `Y` can have the `DynapcnnLayer` containing `C`
    as one of its destinations.

    Parameters
    ----------
        sinabs_edges (list): edges extracted from the computational graph of the network provided to `DynapcnnNetworkGraph` where edges involving
            layers that are ignored (e.g. `nn.Flatten`) have been removed (the nodes previously linked via dropped nodes are linked to each other directly).
        sinabs_modules_map (dict): mapping where the `key` represents a node in the graph and the `value` represents the node's layer.

    Returns
    ----------
        edges_without_merge (list): edges based on `sinabs_edges` but where edges involving a `Merge` layer have been remapped to connect the nodes
            involved in the merging directly.
    """
    edges = copy.deepcopy(sinabs_edges)
    edges_without_merge = []
    merge_nodes = {}

    # finding the nodes representing Merge layers.
    for edge in edges:
        src = edge[0]
        trg = edge[1]

        if isinstance(sinabs_modules_map[src], sinabs.layers.Merge):
            if src not in merge_nodes:
                # found node receiving merged inputs from two previous layers.
                merge_nodes[src] = {'sources': [], 'merge_into': trg}

                for _edge in edges:
                    if _edge[1] == src:
                        # found node used as argument for a Merge layer.
                        merge_nodes[src]['sources'].append(_edge[0])
                        if len(merge_nodes[src]['sources']) > 2:
                            raise ValueError("A Merge layer can not have more than two inputs.")

    for edge in edges:
        # removing edges connection from/to merging layers from the computational graph.
        src = edge[0]
        trg = edge[1]

        if src in merge_nodes:
            # edge (`Merge`, trg) is not necessary for later DynapcnnLayer creation.
            pass
        
        elif trg in merge_nodes:
            # point `src` directly to the node it was previously targeting via a Merge layer.
            new_edge = (src, merge_nodes[trg]['merge_into'])
            edges_without_merge.append(new_edge)

        else:
            # edge not involved in merging.
            edges_without_merge.append(edge)

    return edges_without_merge
    