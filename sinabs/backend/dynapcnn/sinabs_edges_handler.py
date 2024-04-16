# functionality : functions implementing the pre-processing of edges into blocks of nodes (modules) for future creation of DynapcnnLayer objects.
# author        : Willian Soares Girao
# contact       : williansoaresgirao@gmail.com

from typing import Tuple, List
import torch.nn as nn
from .sinabs_edges_utils import *

def process_edge(layers: List[nn.Module], edge: Tuple[int, int], mapper: dict) -> None:
    """ Read in an edge describing the connection between two layers (nodes in the computational graph). If 'edge'
    is a valid connection between two layers, update 'mapper' to incorporate these layers into a new or existing dictonary
    containing the modules comprising a future DynacnnLayer object.

    Parameters
    ----------
        layers  : list of modules returned by 'utils.convert_model_to_layer_list()'.
        edge    : tuple representing the connection between two nodes in computational graph of 'DynapcnnNetworkGraph.snn.spiking_model'.
        mapper  : dictionary where each 'key' is the index of a future 'DynapcnnLayer' and 'value' is a dictionary ('key': node, 'value': module).
    """
    edge_type = is_valid_edge(edge, layers, VALID_SINABS_EDGES)

    if isinstance(edge_type, int):                                  # incorporate modules within the edge to a dict representing a future DynapcnnLayer.
        update_dynapcnnlayer_mapper(edge_type, edge, mapper, layers)
    else:
        raise InvalidEdge(edge, type(layers[edge[0]]), type(layers[edge[1]]))
    
def is_valid_edge(edge: Tuple[int, int], layers: List[nn.Module], valid_edges_map: dict) -> int:
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
    
def update_dynapcnnlayer_mapper(edge_type: int, edge: Tuple[int, int], mapper: dict, layers: List[nn.Module]) -> None:
    """ Parses the nodes within an edge and incorporate them either into a **new** or an **already existing** DynapcnnLayer represented 
    in 'mapper'.
    """

    if edge_type in [0, 6]:
        init_xor_complete_new_dynapcnnlayer_blk(mapper, edge, layers)

    elif edge_type == 1:
        add_pool_to_dynapcnnlayer_blk(mapper, edge, layers)
        
    elif edge_type in [2, 3, 4, 5]:
        connect_dynapcnnlayer_blks(mapper, edge, layers)

    else:
        raise InvalidEdgeType(edge, edge_type)
    
def init_xor_complete_new_dynapcnnlayer_blk(mapper: dict, edge: Tuple[int, int], layers: List[nn.Module]) -> None:
    """ Incorporates nodes from either a '(conv, neuron)' or a '(linear, neuron)' edge. These are either initiating a (new) DynapcnnLayer 
    or completing a conv->neuron sequence (in the case the node for 'conv' as already been incorporated somewhere in 'mapper'). 'nn.Linear' layers 
    are converted into 'nn.Conv2d' by DynapcnnLayer.
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
            mapper[dynapcnnlayer_indx][edge[1]] = layers[edge[1]]
            break
        
    if not matched:                                                 # 'edge[0]' not found: start new DynapcnnLayer block.
        dynapcnnlayer_indx = 0
        for indx, _ in mapper.items():
            dynapcnnlayer_indx += 1
        mapper[dynapcnnlayer_indx] = {edge[0]: layers[edge[0]], edge[1]: layers[edge[1]]}

def connect_dynapcnnlayer_blks(mapper: dict, edge: Tuple[int, int], layers: List[nn.Module]) -> None:
    """ Incorporates nodes from either a '(neuron, conv)/(neuron, lin)' or '(pool, conv)/(pool, lin)' edge. These represent connections between an existing 
    DynapcnnLayer in 'mapper' and a new one yet to be represented in 'mapper'. 'nn.Linear' layers are converted into 'nn.Conv2d' by DynapcnnLayer.
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
            mapper[dynapcnnlayer_indx] = {edge[1]: layers[edge[1]]} # 'edge[1]' starts new DynapcnnLayer block as 'indx+1'.
        else:
            raise UnmatchedNode(edge, node)
    
def add_pool_to_dynapcnnlayer_blk(mapper: dict, edge: Tuple[int, int], layers: List[nn.Module]) -> None:
    """ Incorporating a '(neuron, pool)' edge. Node 'pool' has to be part of an already existing DynapcnnLayer in 'mapper'. """
    matched = False
    for indx, dynapcnnlayer in mapper.items():
        for node, _ in dynapcnnlayer.items():
            if node == edge[0]:
                dynapcnnlayer[edge[1]] = layers[edge[1]]            # 'edge[0]' is a neuron layer inputing into pooling layer 'edge[1]'.
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

def get_dynapcnnlayers_destinations(layers: List[nn.Module], edges: List[Tuple[int, int]], mapper: dict) -> dict:
    """ Loops over the edges list describing the computational graph. It will access each node in the graph and find to which
    DynapcnnLayer they belong to. If source and target belong to different DynapcnnLayers (described as a dictionary in 'mapper')
    the destination of the 'DynapcnnLayer.source' is set to be 'DynapcnnLayer.target'.

    Parameters
    ----------
        layers  : list of modules returned by 'utils.convert_model_to_layer_list()'.
        edges   : list of tuples representing the connection between nodes in computational graph of 'DynapcnnNetworkGraph.snn.spiking_model'.
        mapper  : dictionary where each 'key' is the index of a future 'DynapcnnLayer' and 'value' its modules (output of 'process_edge(mapper)').

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
                
    del used_layer_edges

    return dynapcnnlayers_destinations_map
                
def get_dynapcnnlayer_index(node: int, mapper: dict) -> int:
    """ Returns the DynapcnnLayer index to which 'node' belongs to. """
    for indx, dynapcnnlayer in mapper.items():
        if node in dynapcnnlayer:
            return indx
    raise UnknownNode(node)

def is_valid_dynapcnnlayer_pairing(layers: List[nn.Module], edge: Tuple[int, int], valid_dynapcnnlayer_edges: List[Tuple[nn.Module, nn.Module]]) -> bool:
    """ Checks if the module in 'DynapcnnLayer.source' is targetting a valid module in 'DynapcnnLayer.target'. """
    if (type(layers[edge[0]]), type(layers[edge[1]])) in valid_dynapcnnlayer_edges:
        return True
    else:
        raise InvalidLayerDestination(type(layers[edge[0]]), type(layers[edge[1]]))