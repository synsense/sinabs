"""
functionality : functions implementing the pre-processing of edges into blocks of nodes (modules) for future
                creation of DynapcnnLayer objects.
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

import copy
from typing import Dict, List, Tuple, Type

import torch.nn as nn

import sinabs
import sinabs.layers

from .sinabs_edges_utils import *


def process_edge(
    layers: Dict[int, nn.Module],
    edge: Tuple[int, int],
    mapper: Dict[int, Dict[int, Dict]],
) -> None:
    """Read in an edge describing the connection between two layers (nodes in the computational graph). If `edge`
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
        layers (dict): a dictionary containing the node IDs of the graph as `key` and their associated module as `value`.
        edge (tuple): tuple representing the connection between two nodes in computational graph of 'DynapcnnNetworkGraph.snn.spiking_model'.
        mapper (dict): dictionary where each 'key' is the index of a future 'DynapcnnLayer' and 'value' is a dictionary ('key': node, 'value': module).
    """
    edge_type = get_valid_edge_type(edge, layers, VALID_SINABS_EDGE_TYPE_IDS)

    if edge_type is None:
        raise InvalidEdge(edge, type(layers[edge[0]]), type(layers[edge[1]]))

    # incorporate modules within the edge to a dict representing a future DynapcnnLayer.
    update_dynapcnnlayer_mapper(edge_type, edge, mapper, layers)


def get_valid_edge_type(
    edge: Tuple[int, int],
    layers: Dict[int, nn.Module],
    valid_edge_ids: Dict[Tuple[Type, Type], int],
) -> int:
    """Checks if the modules each node in 'edge' represent are a valid connection between a sinabs network to be
    loaded on Speck and return the edge type

    Parameters
    ----------
        edge (tuple of two int): The edge whose type is to be inferred
        layers (Dict): Dict with node IDs as keys and layer instances as values
        valid_edge_ids: Dict with valid edge-types (tuples of Types) as keys and edge-type-ID as value

    Returns
    ----------
        edge_type: the edge type specified in 'valid_edges_map' ('None' if edge is not valid).
    """
    source_type = type(layers[edge[0]])
    target_type = type(layers[edge[1]])

    return valid_edge_ids.get((source_type, target_type), None)


def update_dynapcnnlayer_mapper(
    edge_type: int,
    edge: Tuple[int, int],
    mapper: Dict[int, Dict[int, Dict]],
    layers: Dict[int, nn.Module],
) -> None:
    """Parses the nodes within an edge and incorporate them either into a **new** or an **already existing** DynapcnnLayer represented
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


def init_xor_complete_new_dynapcnnlayer_blk(
    mapper: Dict[int, Dict[int, Dict]],
    edge: Tuple[int, int],
    layers: Dict[int, nn.Module],
) -> None:
    """Incorporates nodes from either a `(conv, neuron)` or a `(linear, neuron)` edge. These are either initiating a new `dict` mapping
    into a future `DynapcnnLayer` or completing a `conv->neuron` sequence (in the case the node for `conv` as already been incorporated
    somewhere in `mapper`). Obs.: `nn.Linear` layers are converted into `nn.Conv2d` by `DynapcnnLayer`.
    """
    # Search for edge[0] (conv/linear layer) in DynapcnnLayers
    if (dynapcnnlayer_indx := find_initialized_node(edge[0], mapper)) is not None:
        # Add edge[1] (neuron layer) to the same dynapcnn layer
        mapper[dynapcnnlayer_indx][edge[1]] = {
            "layer": layers[edge[1]],
            "input_shape": None,
            "output_shape": None,
        }
    else:
        # Assign new layer, with current length of `mapper` as new unique index
        dynapcnnlayer_indx = len(mapper)
        mapper[dynapcnnlayer_indx] = {
            edge[0]: {
                "layer": layers[edge[0]],
                "input_shape": None,
                "output_shape": None,
            },
            edge[1]: {
                "layer": layers[edge[1]],
                "input_shape": None,
                "output_shape": None,
            },
        }


def connect_dynapcnnlayer_blks(
    mapper: Dict[int, Dict[int, Dict]],
    edge: Tuple[int, int],
    layers: Dict[int, nn.Module],
) -> None:
    """Incorporates nodes from either a `(neuron, conv)/(neuron, lin)` or `(pool, conv)/(pool, lin)` edge. These represent connections between an existing
    `dict` in `mapper` that will be mapped into a `DynapcnnLayer` and a new one yet to be represented in `mapper`. Obs.: `nn.Linear` layers are converted
    into `nn.Conv2d` by `DynapcnnLayer`.
    """
    if find_initialized_node(edge[1], mapper) is None:
        dynapcnnlayer_indx = 0
        matched = False
        for indx, dynapcnnlayer in mapper.items():
            for node, _ in dynapcnnlayer.items():
                if node == edge[0]:  # 'edge[0]' is ending DynapcnnLayer block 'indx'.
                    dynapcnnlayer_indx = indx + 1
                    matched = True
                    break
            if matched:
                break
        if matched:
            while dynapcnnlayer_indx in mapper:
                dynapcnnlayer_indx += 1
            mapper[dynapcnnlayer_indx] = {  # 'edge[1]' starts new DynapcnnLayer block.
                edge[1]: {
                    "layer": layers[edge[1]],
                    "input_shape": None,
                    "output_shape": None,
                }
            }
        else:
            raise UnmatchedNode(edge, node)


def add_pool_to_dynapcnnlayer_blk(
    mapper: Dict[int, Dict[int, Dict]],
    edge: Tuple[int, int],
    layers: Dict[int, nn.Module],
) -> None:
    """Incorporating a `(neuron, pool)` edge. Node `pool` has to be part of an already existing `dict` mapping into a `DynapcnnLaye` in `mapper`."""
    # Search for edge[0] (neuron layer) in DynapcnnLayers
    if (indx := find_initialized_node(edge[0], mapper)) is not None:
        # Add edge[1] (pooling layer) to the same dynapcnn layer
        mapped[indx][edge[1]] = {
            "layer": layers[edge[1]],
            "input_shape": None,
            "output_shape": None,
        }
    else:
        raise UnmatchedNode(edge, node)


def find_initialized_node(node: int, mapper: Dict[int, Dict[int, Dict]]) -> bool:
    """Finds if 'node' existis within 'mapper' and returns layer index."""
    for index, dynapcnnlayer in mapper.items():
        if node in dynapcnnlayer:
            return index
    return None


def get_dynapcnnlayers_destinations(
    layers: Dict[int, nn.Module],
    edges: List[Tuple[int, int]],
    mapper: Dict[int, Dict[int, Dict]],
) -> dict:
    """Loops over the edges list describing the computational graph. It will access each node in the graph and find to which
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

        if source_layer != destination_layer and is_valid_dynapcnnlayer_pairing(
            layers, edge, VALID_DYNAPCNNLAYER_EDGES
        ):
            # valid connection between modules in two different DynapcnnLayer.

            if len(dynapcnnlayers_destinations_map[source_layer]) > 2:
                # DynapcnnLayers can not have more than two destinations.
                raise MaxDestinationsReached(source_layer)
            else:
                if (
                    (destination_layer, source_layer) not in used_layer_edges
                    and destination_layer
                    not in dynapcnnlayers_destinations_map[source_layer]
                ):
                    # edge does not create a loop between layers.
                    dynapcnnlayers_destinations_map[source_layer].append(
                        destination_layer
                    )
                    used_layer_edges.append((source_layer, destination_layer))
                else:
                    raise InvalidLayerLoop(source_layer, destination_layer)

    for dcnnl_idx, destinations in dynapcnnlayers_destinations_map.items():
        # TODO document the 'rescale_factor' better.
        mapper[dcnnl_idx]["destinations"] = destinations
        mapper[dcnnl_idx]["conv_rescale_factor"] = []


def get_dynapcnnlayer_index(node: int, mapper: Dict[int, Dict[int, Dict]]) -> int:
    """Returns the DynapcnnLayer index to which 'node' belongs to."""
    for indx, dynapcnnlayer in mapper.items():
        if node in dynapcnnlayer:
            return indx
    raise UnknownNode(node)


def is_valid_dynapcnnlayer_pairing(
    layers: Dict[int, nn.Module],
    edge: Tuple[int, int],
    valid_dynapcnnlayer_edges: List[Tuple[nn.Module, nn.Module]],
) -> bool:
    """Checks if the module in 'DynapcnnLayer.source' is targetting a valid module in 'DynapcnnLayer.target'."""
    if (type(layers[edge[0]]), type(layers[edge[1]])) in valid_dynapcnnlayer_edges:
        return True
    else:
        raise InvalidLayerDestination(type(layers[edge[0]]), type(layers[edge[1]]))
