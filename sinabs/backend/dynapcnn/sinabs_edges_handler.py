"""
functionality : functions implementing the pre-processing of edges into blocks of nodes (modules) for future
                creation of DynapcnnLayer objects.
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

from collections import deque
from typing import Dict, List, Set, Tuple, Type

from torch import Size, nn

from .connectivity_specs import VALID_SINABS_EDGE_TYPES
from .exceptions import InvalidEdge, UnmatchedNode, UnmatchedPoolingEdges
from .utils import Edge


def collect_dynapcnn_layer_info(
    indx_2_module_map: Dict[int, nn.Module],
    edges: Set[Edge],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
) -> Dict[int, Dict]:
    """Collect information to construct DynapcnnLayer instances.

    Validate and sort edges based on the type of nodes they connect.
    Iterate over edges in order of their type. For each neuron->weight edge
    generate a new dict to collect information for the corresponding dynapcnn layer.
    Then add pooling based on neuron->pooling type edges. Collect additional pooling
    from pooling->pooling type edges. Finally set layer destinations based on
    neuron/pooling->weight type of edges.

    Parameters
    ----------
    indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`
    edges (set of tuples): Represent connections between two nodes in computational graph
    nodes_io_shapes (dict): Map from node ID to dict containing node's in- and output shapes

    Returns
    -------
    dynapcnn_layer_info (dict): Each 'key' is the index of a future 'DynapcnnLayer' and
        'value' is a dictionary, with keys 'conv', 'neuron', and 'destinations',
        containing corresponding node ids and modules required to build the layer
    """
    # TODO: Handle DVS layer

    # Sort edges by edge type (type of layers they connect)
    edges_by_type: Dict[str, Set[Edge]] = dict()
    for edge in edges:
        edge_type = get_valid_edge_type(
            edge, indx_2_module_map, VALID_SINABS_EDGE_TYPES
        )

        # Validate edge type
        if edge_type is None:
            raise InvalidEdge(
                edge, type(indx_2_module_map[edge[0]]), type(indx_2_module_map[edge[1]])
            )

        if edge_type in edges_by_type:
            edges_by_type[edge_type].add(edge)
        else:
            edges_by_type[edge_type] = {edge}

    # Dict to collect information for each future dynapcnn layer
    dynapcnn_layer_info = dict()
    # Map node IDs to dynapcnn layer ID
    node_2_layer_map = dict()

    # Each weight->neuron connection instantiates a new, unique dynapcnn layer
    while edges_by_type["weight-neuron"]:
        edge = edges_by_type["weight-neuron"].pop()
        init_new_dynapcnnlayer_entry(
            dynapcnn_layer_info,
            edge,
            indx_2_module_map,
            nodes_io_shapes,
            node_2_layer_map,
        )

    # "pooling-pooling" edges are optional. Unlike other types, missing entry would cause exception.
    # Therefore add empty set if not existing
    if "pooling-pooling" not in edges_by_type:
        edges_by_type["pooling-pooling"] = set()

    # Add pooling based on neuron->pooling connections
    while edges_by_type["neuron-pooling"]:
        edge = edges_by_type["neuron-pooling"].pop()
        # Search pooling-pooling edges for chains of pooling and add to existing entry
        pooling_chains, edges_used = trace_paths(
            edge[1], edges_by_type["pooling-pooling"]
        )
        add_pooling_to_entry(
            dynapcnn_layer_info,
            edge,
            pooling_chains,
            indx_2_module_map,
            node_2_layer_map,
        )
        # Remove handled pooling-pooling edges
        edges_by_type["pooling-pooling"].difference_update(edges_used)
    # After adding pooling make sure all pooling-pooling edges have been handled
    if len(edges_by_type["pooling-pooling"]) > 0:
        raise UnmatchedPoolingEdges(edges_by_type["pooling-pooling"])

    # Process all edges connecting two dynapcnn layers
    while edges_by_type["neuron-weight"]:
        edge = edges_by_type["neuron-weight"].pop()
        set_neuron_layer_destination(dynapcnn_layer_info, edge, node_2_layer_map)

    while edges_by_type["pooling-weight"]:
        edge = edges_by_type["pooling-weight"].pop()
        set_pooling_layer_destination(dynapcnn_layer_info, edge, node_2_layer_map)

    # Make sure we have taken care of all edges
    assert all(len(edges) == 0 for edges in edges_by_type.values())

    return dynapcnn_layer_info


def get_valid_edge_type(
    edge: Edge,
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


def init_new_dynapcnnlayer_entry(
    dynapcnn_layer_info: Dict[int, Dict[int, Dict]],
    edge: Edge,
    indx_2_module_map: Dict[int, nn.Module],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    node_2_layer_map: Dict[int, int],
) -> None:
    """Initiate dict to hold information for new dynapcnn layer based on a "weight->neuron" edge.
    Change `dynapcnn_layer_info` in-place.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    edge: Tuple of 2 integers, indicating edge between two nodes in graph.
        Edge source has to be within an existing entry of `dynapcnn_layer_info`.
    indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`
    nodes_io_shapes (dict): Map from node ID to dict containing node's in- and output shapes
    node_2_layer_map (dict): Maps each node ID to the ID of the layer it is assigned to.
        Will be updated in-place.
    """
    # Make sure there are no existing entries holding any of the modules connected by `edge`
    assert edge[0] not in node_2_layer_map
    assert edge[1] not in node_2_layer_map

    # Take current length of the dict as new, unique ID
    layer_id = len(dynapcnn_layer_info)
    assert layer_id not in dynapcnn_layer_info

    dynapcnn_layer_info[layer_id] = {
        "input_shape": nodes_io_shapes[edge[0]],
        "conv": {
            "module": indx_2_module_map[edge[0]],
            "node_id": edge[0],
        },
        "neuron": {
            "module": indx_2_module_map[edge[1]],
            "node_id": edge[1],
        },
        # This will be used later to account for average pooling in preceding layers
        "rescale_factors": {},
    }
    node_2_layer_map[edge[0]] = layer_id
    node_2_layer_map[edge[1]] = layer_id


def add_pooling_to_entry(
    dynapcnn_layer_info: Dict[int, Dict[int, Dict]],
    edge: Edge,
    pooling_chains: List[deque[int]],
    indx_2_module_map: Dict[int, nn.Module],
    node_2_layer_map: Dict[int, int],
) -> None:
    """Add or extend destination information to existing entry in `dynapcnn_layer_info`.

    Correct entry is identified by existing neuron node. Destination information is a
    dict containing list of IDs and list of modules for each chains of pooling nodes.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    edge: Tuple of 2 integers, indicating edge between two nodes in graph.
        Edge source has to be within an existing entry of `dynapcnn_layer_info`.
    pooling_chains: List of deque of int. All sequences ("chains") of connected pooling nodes,
        starting from edge[1]
    indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`
    node_2_layer_map (dict): Maps each node ID to the ID of the layer it is assigned to.
        Will be updated in-place.
    """
    # Find layer containing edge[0]
    try:
        layer_idx = node_2_layer_map[edge[0]]
    except KeyError:
        raise UnmatchedNode(edge, edge[0])
    # Make sure all pooling chains start with expected node
    assert all(chain[0] == edge[1] for chain in pooling_chains)

    # Layer entry might already have `destinations` key (if neuron layer has fanout > 1)
    layer_info = dynapcnn_layer_info[layer_idx]
    if "destinations" not in layer_info:
        layer_info["destinations"] = []

    # Keep track of all nodes that have been added
    new_nodes = set()

    # For each pooling chain initialize new destination
    for chain in pooling_chains:
        layer_info["destinations"].append(
            {
                "pooling_ids": chain,
                "pooling_modules": [indx_2_module_map[idx] for idx in chain],
            }
        )
        new_nodes.update(set(chain))

    for node in new_nodes:
        # Make sure new pooling nodes have not been used elsewhere
        assert node not in node_2_layer_map
        node_2_layer_map[node] = layer_idx


def set_neuron_layer_destination(
    dynapcnn_layer_info: Dict[int, Dict[int, Dict]],
    edge: Edge,
    node_2_layer_map: Dict[int, int],
) -> None:
    """Set destination layer without pooling.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    edge: Tuple of 2 integers, indicating edge between two nodes in graph.
        Edge source has to be within an existing entry of `dynapcnn_layer_info`.
    node_2_layer_map (dict): Maps each node ID to the ID of the layer it is assigned to.
        Will be updated in-place.
    """
    # Make sure both source (neuron layer) and target (weight layer) have been previously processed
    try:
        source_layer_idx = node_2_layer_map[edge[0]]
    except KeyError:
        raise UnmatchedNode(edge, edge[0])
    try:
        destination_layer_idx = node_2_layer_map[edge[1]]
    except KeyError:
        raise UnmatchedNode(edge, edge[1])

    # Source layer entry might already have `destinations` key (if neuron layer has fanout > 1)
    layer_info = dynapcnn_layer_info[source_layer_idx]
    if "destinations" not in layer_info:
        layer_info["destinations"] = []

    # Add new destination
    layer_info["destinations"].append(
        {
            "pooling_ids": [],
            "pooling_modules": [],
            "destination_layer": destination_layer_idx,
        }
    )


def set_pooling_layer_destination(
    dynapcnn_layer_info: Dict[int, Dict[int, Dict]],
    edge: Edge,
    node_2_layer_map: Dict[int, int],
) -> None:
    """Set destination layer with pooling.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    edge: Tuple of 2 integers, indicating edge between two nodes in graph.
        Edge source has to be within an existing entry of `dynapcnn_layer_info`.
    node_2_layer_map (dict): Maps each node ID to the ID of the layer it is assigned to.
        Will be updated in-place.
    """
    # Make sure both source (pooling layer) and target (weight layer) have been previously processed
    try:
        source_layer_idx = node_2_layer_map[edge[0]]
    except KeyError:
        raise UnmatchedNode(edge, edge[0])
    try:
        destination_layer_idx = node_2_layer_map[edge[1]]
    except KeyError:
        raise UnmatchedNode(edge, edge[1])

    # Source layer entry should already have `destinations` key
    layer_info = dynapcnn_layer_info[source_layer_idx]

    # Find current source node within destinations
    matched = False
    for destination in layer_info["destinations"]:
        if destination["pooling_ids"][-1] == edge[0]:
            matched = True
            break
    if not matched:
        raise UnmatchedNode(edge, edge[0])

    # Set destination layer within destination dict that holds current source node
    destination["destination_layer"] = destination_layer_idx


def trace_paths(node: int, remaining_edges: Set[Edge]) -> List[deque[int]]:
    """Trace any path of collected edges through the graph.

    Start with `node`, and recursively look for paths of connected nodes
    within `remaining edges.`

    Parameters
    ----------
    node (int): ID of current node
    remaining_edges: Set of remaining edges still to be searched

    Returns
    -------
    paths: List of deque of int, all paths of connected edges starting from `node`.
    processed_edges: Set of edges that are part of the returned paths
    """
    paths = []
    processed_edges = set()
    for src, tgt in remaining_edges:
        if src == node:
            processed_edges.add((src, tgt))
            # For each edge with `node` as source, find subsequent pooling nodes recursively
            new_remaining = remaining_edges.difference({(src, tgt)})
            branches, new_processed = trace_paths(tgt, new_remaining)
            # Make sure no edge was processed twice
            assert len(processed_edges.intersection(new_processed)) == 0

            # Keep track of newly processed edges
            processed_edges.update(new_processed)

            # Collect all branching paths of pooling, inserting src at beginning
            for branch in branches:
                branch.appendleft(src)
                paths.append(branch)

    if not paths:
        # End of recursion: instantiate a deque only with node
        paths = [deque([node])]

    return paths, processed_edges
