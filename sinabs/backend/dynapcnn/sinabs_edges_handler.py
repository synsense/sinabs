"""
functionality : functions implementing the pre-processing of edges into blocks of nodes (modules) for future
                creation of DynapcnnLayer objects.
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

from collections import deque
from typing import Dict, List, Optional, Set, Tuple, Type, Union

from torch import Size, nn

from sinabs.layers import SumPool2d
from sinabs.utils import expand_to_pair

from .connectivity_specs import VALID_SINABS_EDGE_TYPES, Pooling
from .crop2d import Crop2d
from .dvs_layer import DVSLayer
from .exceptions import (
    InvalidEdge,
    InvalidGraphStructure,
    default_invalid_structure_string,
)
from .flipdims import FlipDims
from .utils import Edge, merge_bn


def remap_edges_after_drop(
    dropped_node: int, source_of_dropped_node: int, edges: Set[Edge]
) -> Set[Edge]:
    """Creates a new set of edges from `edges`. All edges where `dropped_node` is the source node will be used to generate
    a new edge where `source_of_dropped_node` becomes the source node (the target is kept).

    Parameters
    ----------
    - dropped_node (int):
    - source_of_dropped_node (int):
    - edges (set): tuples describing the connections between layers in `spiking_model`.

    Returns
    -------
    - remapped_edges (set): new set of edges with `source_of_dropped_node` as the source node where `dropped_node` used to be.
    """
    remapped_edges = set()

    for src, tgt in edges:
        if src == dropped_node:
            remapped_edges.add((source_of_dropped_node, tgt))

    return remapped_edges


def handle_batchnorm_nodes(
    edges: Set[Edge],
    indx_2_module_map: Dict[int, nn.Module],
    name_2_indx_map: Dict[str, int],
) -> None:
    """Merges `BatchNorm2d`/`BatchNorm1d` layers into `Conv2d`/`Linear` ones. The batch norm nodes will be removed from the graph (by updating all variables
    passed as arguments in-place) after their properties are used to re-scale the weights of the convolutional/linear layers associated with batch
    normalization via the `weight-batchnorm` edges found in the original graph.

    Parameters
    ----------
    - edges (set): tuples describing the connections between layers in `spiking_model`.
    - indx_2_module_map (dict): the mapping between a node (`key` as an `int`) and its module (`value` as a `nn.Module`).
    - name_2_indx_map (dict): Map from node names to unique indices.
    """

    # Gather indexes of the BatchNorm2d/BatchNorm1d nodes.
    bnorm_nodes = {
        index
        for index, module in indx_2_module_map.items()
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d))
    }

    if len(bnorm_nodes) == 0:
        # There are no edges with batch norm - nothing to do here.
        return

    # Find weight-bnorm edges.
    weight_bnorm_edges = {
        (src, tgt)
        for (src, tgt) in edges
        if (
            isinstance(indx_2_module_map[src], nn.Conv2d)
            and isinstance(indx_2_module_map[tgt], nn.BatchNorm2d)
        )
        or (
            isinstance(indx_2_module_map[src], nn.Linear)
            and isinstance(indx_2_module_map[tgt], nn.BatchNorm1d)
        )
    }

    # Merge conv/linear and bnorm layers using 'weight-bnorm' edges.
    for edge in weight_bnorm_edges:
        bnorm = indx_2_module_map[edge[1]]
        weight = indx_2_module_map[edge[0]]

        # merge and update weight node.
        indx_2_module_map[edge[0]] = merge_bn(weight, bnorm)

    # Point weight nodes to the targets of their respective batch norm nodes.
    new_edges = set()
    for weight, bnorm in weight_bnorm_edges:
        new_edges.update(
            remap_edges_after_drop(
                dropped_node=bnorm, source_of_dropped_node=weight, edges=edges
            )
        )

    # Remove references to the bnorm node.

    for idx in bnorm_nodes:
        indx_2_module_map.pop(idx)

    for name in [name for name, indx in name_2_indx_map.items() if indx in bnorm_nodes]:
        name_2_indx_map.pop(name)

    for edge in weight_bnorm_edges:
        edges.remove(edge)

    for edge in [
        (src, tgt) for (src, tgt) in edges if (src in bnorm_nodes or tgt in bnorm_nodes)
    ]:
        edges.remove(edge)

    # Update 'edges' in-place to incorporate new edges:
    for edge in new_edges:
        edges.add(edge)


def fix_dvs_module_edges(
    edges: Set[Edge],
    indx_2_module_map: Dict[int, nn.Module],
    name_2_indx_map: Dict[str, int],
    entry_nodes: Set[Edge],
) -> None:
    """All arguments are modified in-place to fix wrong node extractions from NIRtorch when a DVSLayer istance is the first layer in the network.

    Modifies `edges` to re-structure the edges related witht the DVSLayer instance. The DVSLayer's forward method feeds data in the
    sequence 'DVS -> DVS.pool -> DVS.crop -> DVS.flip', so we remove edges involving these nodes (that are internaly implementend in
    the DVSLayer) from the graph and point the node of DVSLayer to the node where it should send its output to. This is also removes
    a self-recurrent node with edge '(FlipDims, FlipDims)' that is wrongly extracted.

    Modifies `indx_2_module_map` and `name_2_indx_map` to remove the internal DVSLayer nodes (Crop2d, FlipDims and DVSLayer's pooling) since
    these should not be independent nodes in the graph.

    Modifies `entry_nodes` such that the DVSLayer becomes the only entry node of the graph.

    Parameters
    ----------
    - edges (set): tuples describing the connections between layers in `spiking_model`.
    - indx_2_module_map (dict): the mapping between a node (`key` as an `int`) and its module (`value` as a `nn.Module`).
    - name_2_indx_map (dict): Map from node names to unique indices.
    - entry_nodes (set): IDs of nodes acting as entry points for the network (i.e., receiving external input).
    """
    # TODO - the 'fix_' is to imply there's something odd with the extracted adges for the forward pass implemented by
    # the DVSLayer. For now this function is fixing these edges to have them representing the information flow through
    # this layer as **it should be** but the graph tracing of NIR should be looked into to solve the root problem.

    # spot nodes (ie, modules) used in a DVSLayer instance's forward pass (including the DVSLayer node itself).
    dvslayer_nodes = {
        index: module
        for index, module in indx_2_module_map.items()
        if any(
            isinstance(module, dvs_node) for dvs_node in (DVSLayer, Crop2d, FlipDims)
        )
    }

    if len(dvslayer_nodes) <= 1:
        # No module within the DVSLayer instance appears as an independent node - nothing to do here.
        return

    # TODO - a `SumPool2d` is also a node that's used inside a DVSLayer instance. In what follows we try to find it
    # by looking for pooling nodes that appear in a (pool, crop) edge - the assumption being that if the pooling is
    # inputing into a crop layer than the pool is inside the DVSLayer instance. It feels like a hacky way to do it
    # so we should revise this.
    dvslayer_nodes.update(
        {
            edge[0]: indx_2_module_map[edge[0]]
            for edge in edges
            if isinstance(indx_2_module_map[edge[0]], SumPool2d)
            and isinstance(indx_2_module_map[edge[1]], Crop2d)
        }
    )

    # NIR is extracting an edge (FlipDims, FlipDims) from the DVSLayer: remove self-recurrent nodes from the graph.
    for edge in [
        (src, tgt)
        for (src, tgt) in edges
        if (src == tgt and isinstance(indx_2_module_map[src], FlipDims))
    ]:
        edges.remove(edge)

    # Since NIR is not extracting the edges for the DVSLayer correctly, remove all edges involving the DVS.
    for edge in [
        (src, tgt)
        for (src, tgt) in edges
        if (src in dvslayer_nodes or tgt in dvslayer_nodes)
    ]:
        edges.remove(edge)

    # Get node's indexes based on the module type - just for validation.
    dvs_node = [
        key for key, value in dvslayer_nodes.items() if isinstance(value, DVSLayer)
    ]
    dvs_pool_node = [
        key for key, value in dvslayer_nodes.items() if isinstance(value, SumPool2d)
    ]
    dvs_crop_node = [
        key for key, value in dvslayer_nodes.items() if isinstance(value, Crop2d)
    ]
    dvs_flip_node = [
        key for key, value in dvslayer_nodes.items() if isinstance(value, FlipDims)
    ]

    if any(
        len(node) > 1
        for node in [dvs_node, dvs_pool_node, dvs_crop_node, dvs_flip_node]
    ):
        raise ValueError(
            f"Internal DVS nodes should be single instances but multiple have been found: dvs_node: {len(dvs_node)} dvs_pool_node: {len(dvs_pool_node)} dvs_crop_node: {len(dvs_crop_node)} dvs_flip_node: {len(dvs_flip_node)}"
        )

    # Remove dvs_pool, dvs_crop and dvs_flip nodes from `indx_2_module_map` (these operate within the DVS, not as independent nodes of the final graph).
    indx_2_module_map.pop(dvs_pool_node[-1])
    indx_2_module_map.pop(dvs_crop_node[-1])
    indx_2_module_map.pop(dvs_flip_node[-1])

    # Remove internal DVS modules from name/index map.
    # Iterate over copy to prevent iterable from changing size.
    n2i_map_copy = {k: v for k, v in name_2_indx_map.items()}
    for name, index in n2i_map_copy.items():
        if index in [dvs_pool_node[-1], dvs_crop_node[-1], dvs_flip_node[-1]]:
            name_2_indx_map.pop(name)

    dvs_node = dvs_node[0]
    if edges:
        # Add edges from 'dvs' node to the entry point of the graph.
        all_sources, all_targets = zip(*edges)
        local_entry_nodes = set(all_sources) - set(all_targets)
        edges.update({(dvs_node, node) for node in local_entry_nodes})

    # DVS becomes the only entry node of the graph.
    entry_nodes.clear()
    entry_nodes.add(dvs_node)


def merge_dvs_pooling_edge(
    edges: Set[Edge],
    indx_2_module_map: Dict[int, nn.Module],
    name_2_indx_map: Dict[str, int],
) -> None:
    """If a 'dvs-pooling' edge existis, the pooling is incorporated into the DVSLayer node if `DVSLayer.pool_layer` has
    default values. All arguments are modified in-place to remove the references to the incorporated pooling node.

    Parameters
    ----------
    - edges (set): tuples describing the connections between layers in `spiking_model`.
    - indx_2_module_map (dict): the mapping between a node (`key` as an `int`) and its module (`value` as a `nn.Module`).
    - name_2_indx_map (dict): Map from node names to unique indices.
    """
    # Find 'DVSLayer-pooling' edge.
    dvs_pool_edge = [
        (src, tgt)
        for (src, tgt) in edges
        if (
            isinstance(indx_2_module_map[src], DVSLayer)
            and any(isinstance(indx_2_module_map[tgt], pool) for pool in Pooling)
        )
    ]

    if len(dvs_pool_edge) == 0:
        # No dvs-pooling edge exists - nothing to do here.
        return
    if len(dvs_pool_edge) > 1:
        # DVSLayer in the original network can have only a single pooling layer liked to it.
        raise ValueError(
            f"DVSLayer has multiple outgoing edges to pooling layers: {dvs_pool_edge}. "
            "Unlike convolutional layers, for DVS layers, pooling is set globally for "
            "all destinations. Therefore a DVSLayer can be followed by at most one "
            "pooling layer."
        )

    (dvs_indx, pool_indx) = dvs_pool_edge[-1]
    dvs_layer = indx_2_module_map[dvs_indx]
    pool_layer = indx_2_module_map[pool_indx]

    # Checking pooling can be incorporated into the DVSLayer.
    if expand_to_pair(dvs_layer.pool_layer.kernel_size) != [1, 1] or expand_to_pair(
        dvs_layer.pool_layer.stride
    ) != [1, 1]:
        raise ValueError(
            "DVSLayer with pooling is followed by another pooling layer. "
            "This is currently not supported. Please update the network "
            "such that all pooling is either done by the DVSLayer or by "
            "the following pooling layer."
        )
    crop_layer = dvs_layer.crop_layer
    if (
        crop_layer.top_crop != 0
        or crop_layer.left_crop != 0
        or crop_layer.bottom_crop != dvs_layer.input_shape[1]
        or crop_layer.right_crop != dvs_layer.input_shape[2]
    ):
        raise ValueError(
            "DVSLayer with cropping is followed by a pooling layer. "
            "This is currently not supported. Please define pooling "
            "directly within the DVSLayer (with the `pool` argument) "
            "and remove the pooling layer that follows the DVSLayer"
        )
    flip_layer = dvs_layer.flip_layer
    if flip_layer.flip_x or flip_layer.flip_y or flip_layer.swap_xy:
        raise ValueError(
            "DVSLayer with flipping or dimension swapping is followed "
            "by a pooling layer. This is currently not supported. "
            "Please define pooling directly within the DVSLayer "
            "(with the `pool` argument) and remove the pooling "
            "layer that follows the DVSLayer"
        )

    # Set DVSLayer.pool to have same config. as the independent pooling layer.
    dvs_layer.pool_layer.kernel_size = pool_layer.kernel_size
    dvs_layer.pool_layer.stride = pool_layer.stride

    # Pooling incorporated to the DVSLayer: remove its trace from mappings.
    indx_2_module_map.pop(pool_indx)
    name_2_indx_map.pop(
        [name for name, indx in name_2_indx_map.items() if indx == pool_indx][-1]
    )

    # Since pool is part of the DVSLayer we now make edges where pool was a source to have DVSLayer as a source.
    for edge in [edge for edge in edges if edge[0] == pool_indx]:
        edges.remove(edge)
        edges.update({(dvs_indx, edge[1])})

    # Remove original 'dvs-pool' edge.
    edges.remove((dvs_indx, pool_indx))

    # Checks if any traces of the original pooling node can still be found.
    if (
        len([edge for edge in edges if (edge[0] == pool_indx or edge[1] == pool_indx)])
        != 0
    ):
        raise ValueError(
            "Edges involving the pooling layer merged into the DVSLayer are still present in the graph."
        )


def collect_dynapcnn_layer_info(
    indx_2_module_map: Dict[int, nn.Module],
    edges: Set[Edge],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    entry_nodes: Set[int],
) -> Tuple[Dict[int, Dict], Union[Dict, None]]:
    """Collect information to construct DynapcnnLayer instances.

    Validate and sort edges based on the type of nodes they connect.
    Iterate over edges in order of their type. For each neuron->weight edge
    generate a new dict to collect information for the corresponding dynapcnn layer.
    Then add pooling based on neuron->pooling type edges. Collect additional pooling
    from pooling->pooling type edges. Finally set layer destinations based on
    neuron/pooling->weight type of edges.

    Parameters
    ----------
    - indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`
    - edges (set of tuples): Represent connections between two nodes in computational graph
    - nodes_io_shapes (dict): Map from node ID to dict containing node's in- and output shapes
    - entry_nodes (set of int): IDs of nodes that receive external input

    Returns
    -------
    dynapcnn_layer_info (dict): Each 'key' is the index of a future 'DynapcnnLayer' and
        'value' is a dictionary, with keys 'conv', 'neuron', and 'destinations',
        containing corresponding node ids and modules required to build the layer
    dvs_layer_info (dict or None): If a DVSLayer is part of the network, this will
        be a dict containing the layer itself and its destination indices.
    """

    # Sort edges by edge type (type of layers they connect)
    edges_by_type: Dict[str, Set[Edge]] = sort_edges_by_type(
        edges=edges, indx_2_module_map=indx_2_module_map
    )
    edge_counts_by_type = {t: len(e) for t, e in edges_by_type.items()}

    # Dict to collect information for each future dynapcnn layer
    dynapcnn_layer_info = dict()
    dvs_layer_info = None
    # Map node IDs to dynapcnn layer ID
    node_2_layer_map = dict()

    # Each weight->neuron connection instantiates a new, unique dynapcnn layer
    weight_neuron_edges = edges_by_type.get("weight-neuron", set())
    while weight_neuron_edges:
        edge = weight_neuron_edges.pop()
        init_new_dynapcnnlayer_entry(
            dynapcnn_layer_info,
            edge,
            indx_2_module_map,
            nodes_io_shapes,
            node_2_layer_map,
            entry_nodes,
        )

    # Process all dvs->weight edges connecting the DVS camera to a unique dynapcnn layer.
    dvs_weight_edges = edges_by_type.get("dvs-weight", set())
    while dvs_weight_edges:
        edge = dvs_weight_edges.pop()
        dvs_layer_info = add_or_update_dvs_to_entry(
            edge,
            dvs_layer_info,
            indx_2_module_map,
            node_2_layer_map,
            nodes_io_shapes,
        )

    # Process all edges connecting two dynapcnn layers that do not include pooling
    neuron_weight_edges = edges_by_type.get("neuron-weight", set())
    while neuron_weight_edges:
        edge = neuron_weight_edges.pop()
        set_neuron_layer_destination(
            dynapcnn_layer_info,
            edge,
            node_2_layer_map,
            nodes_io_shapes,
            indx_2_module_map,
        )

    # Add pooling based on neuron->pooling connections
    pooling_pooling_edges = edges_by_type.get("pooling-pooling", set())
    neuron_pooling_edges = edges_by_type.get("neuron-pooling", set())
    while neuron_pooling_edges:
        edge = neuron_pooling_edges.pop()
        # Search pooling-pooling edges for chains of pooling and add to existing entry
        pooling_chains, edges_used = trace_paths(edge[1], pooling_pooling_edges)
        add_pooling_to_entry(
            dynapcnn_layer_info,
            edge,
            pooling_chains,
            indx_2_module_map,
            node_2_layer_map,
        )
        # Remove handled pooling-pooling edges
        pooling_pooling_edges.difference_update(edges_used)

    # After adding pooling make sure all pooling-pooling edges have been handled
    if len(pooling_pooling_edges) > 0:
        unmatched_layers = {edge[0] for edge in pooling_pooling_edges}
        raise InvalidGraphStructure(
            f"Pooling layers {unmatched_layers} could not be assigned to a "
            "dynapcnn layer. This is likely due to an unsupported SNN "
            "architecture. Pooling layers must always be preceded by a "
            "spiking layer (`IAFSqueeze`), another pooling layer, or"
            "DVS input"
        )

    # Add all edges connecting pooling to a new dynapcnn layer
    pooling_weight_edges = edges_by_type.get("pooling-weight", set())
    while pooling_weight_edges:
        edge = pooling_weight_edges.pop()
        set_pooling_layer_destination(
            dynapcnn_layer_info,
            edge,
            node_2_layer_map,
            nodes_io_shapes,
            indx_2_module_map,
        )

    # Make sure we have taken care of all edges
    assert all(len(edges) == 0 for edges in edges_by_type.values())

    # Set minimal destination entries for layers without child nodes, to act as network outputs
    set_exit_destinations(dynapcnn_layer_info)

    # Assert formal correctness of layer info
    verify_layer_info(dynapcnn_layer_info, edge_counts_by_type)

    return dynapcnn_layer_info, dvs_layer_info


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


def sort_edges_by_type(
    edges: Set[Edge], indx_2_module_map: Dict[int, Type]
) -> Dict[str, Set[Edge]]:
    """Sort edges by the type of nodes they connect

    Parameters
    ----------
    edges (set of tuples): Represent connections between two nodes in computational graph
    indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`

    Returns
    -------
    Dict with possible keys "weight-neuron", "neuron-weight", "neuron-pooling", "pooling-pooling",
        and "pooling-weight". Values are sets of edges corresponding to these types.
    """
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

    return edges_by_type


def init_new_dynapcnnlayer_entry(
    dynapcnn_layer_info: Dict[int, Dict[int, Dict]],
    edge: Edge,
    indx_2_module_map: Dict[int, nn.Module],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    node_2_layer_map: Dict[int, int],
    entry_nodes: Set[int],
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
    entry_nodes (set of int): IDs of nodes that receive external input
    """
    # Make sure there are no existing entries holding any of the modules connected by `edge`
    assert edge[0] not in node_2_layer_map
    assert edge[1] not in node_2_layer_map

    # Take current length of the dict as new, unique ID
    layer_id = len(dynapcnn_layer_info)
    assert layer_id not in dynapcnn_layer_info

    dynapcnn_layer_info[layer_id] = {
        "input_shape": nodes_io_shapes[edge[0]]["input"],
        "conv": {
            "module": indx_2_module_map[edge[0]],
            "node_id": edge[0],
        },
        "neuron": {
            "module": indx_2_module_map[edge[1]],
            "node_id": edge[1],
        },
        # This will be used later to account for average pooling in preceding layers
        "rescale_factors": set(),
        "is_entry_node": edge[0] in entry_nodes,
        # Will be populated by `set_[pooling/neuron]_layer_destination`
        "destinations": [],
    }
    node_2_layer_map[edge[0]] = layer_id
    node_2_layer_map[edge[1]] = layer_id


def add_pooling_to_entry(
    dynapcnn_layer_info: Dict[int, Dict],
    edge: Edge,
    pooling_chains: List[deque[int]],
    indx_2_module_map: Dict[int, nn.Module],
    node_2_layer_map: Dict[int, int],
) -> None:
    """Add or extend destination information with pooling for existing
    entry in `dynapcnn_layer_info`.

    Correct entry is identified by existing neuron node. Destination information is a
    dict containing list of IDs and list of modules for each chains of pooling nodes.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    edge: Tuple of 2 integers, indicating edge between a neuron node and the pooling
        node that starts all provided `pooling_chains`.
        Edge source has to be a neuron node within an existing entry of
        `dynapcnn_layer_info`, i.e. it has to have been processed already.
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
        neuron_layer = indx_2_module_map[edge[0]]
        raise InvalidGraphStructure(
            f"Spiking layer {neuron_layer} cannot be assigned to a dynapcnn layer. "
            "This is likely due to an unsupported SNN architecture. Spiking "
            "layers have to be preceded by a weight layer (`nn.Conv2d` or "
            "`nn.Linear`)."
        )
    # Make sure all pooling chains start with expected node
    assert all(chain[0] == edge[1] for chain in pooling_chains)

    # Keep track of all nodes that have been added
    new_nodes = set()

    # For each pooling chain initialize new destination
    layer_info = dynapcnn_layer_info[layer_idx]
    for chain in pooling_chains:
        layer_info["destinations"].append(
            {
                "pooling_ids": chain,
                "pooling_modules": [indx_2_module_map[idx] for idx in chain],
                # Setting `destination_layer` to `None` allows for this layer
                # to act as network exit point if not destination is added later
                "destination_layer": None,
            }
        )
        new_nodes.update(set(chain))

    for node in new_nodes:
        # Make sure new pooling nodes have not been used elsewhere
        assert node not in node_2_layer_map
        node_2_layer_map[node] = layer_idx


def add_or_update_dvs_to_entry(
    edge: Edge,
    dvs_layer_info: Union[None, Dict],
    indx_2_module_map: Dict[int, nn.Module],
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
) -> Dict:
    """Initiate or update dict to hold information for a DVS Layer configuration based on a "dvs-weight" edges.
    Change `dynapcnn_layer_info` in-place. If a entry for the DVS node exists the function will add a new entry
    to the `desctinations` key of its dictionary.

    Parameters
    ----------
    edge: Tuple of 2 integers, indicating edge between two nodes in graph.
        Edge target has to be within an existing entry of `dynapcnn_layer_info`.
    dvs_layer_info: Dict containing information about the DVSLayer. If `None`,
        will instantiate a new dict
    indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`
    node_2_layer_map (dict): Maps each node ID to the ID of the layer it is assigned to.
        Will be updated in-place.
    nodes_io_shapes (dict): Map from node ID to dict containing node's in- and output shapes
    """

    # This should never happen
    assert isinstance(
        indx_2_module_map[edge[0]], DVSLayer
    ), f"Source node in edge {edge} is of type {type(DVSLayer)} (it should be a DVSLayer instance)."

    # Find destination layer index
    try:
        destination_layer_idx = node_2_layer_map[edge[1]]
    except KeyError:
        weight_layer = indx_2_module_map[edge[1]]
        raise InvalidGraphStructure(
            f"Weight layer {weight_layer} cannot be assigned to a dynapcnn layer. "
            "This is likely due to an unsupported SNN architecture. Weight "
            "layers have to be followed by a spiking layer (`sl.IAFSqueeze`)."
        )

    if dvs_layer_info is None:
        # DVS node hasn't been initialized yet
        # Init. entry for a DVS layer using its configuration dict.
        dvs_layer_info = {
            "node_id": edge[0],
            "input_shape": nodes_io_shapes[edge[0]]["input"],
            "module": indx_2_module_map[edge[0]],
            "destinations": [node_2_layer_map[edge[1]]],
        }

        node_2_layer_map[edge[0]] = "dvs"
    else:
        # Update entry for DVS with new destination.
        assert dvs_layer_info["node_id"] == edge[0]
        assert destination_layer_idx not in dvs_layer_info["destinations"]

        dvs_layer_info["destinations"].append(destination_layer_idx)

    return dvs_layer_info


def set_exit_destinations(dynapcnn_layer: Dict) -> None:
    """Set minimal destination entries for layers that don't have any.

    This ensures that the forward methods of the resulting DynapcnnLayer
    instances return an output, letting these layers act as exit points
    of the network.
    The destination layer will be `None`, and no pooling applied.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    """
    for layer_info in dynapcnn_layer.values():
        if not (destinations := layer_info["destinations"]):
            # Add `None` destination to empty destination lists
            destinations.append(
                {
                    "pooling_ids": [],
                    "pooling_modules": [],
                    "destination_layer": None,
                }
            )


def set_neuron_layer_destination(
    dynapcnn_layer_info: Dict[int, Dict],
    edge: Edge,
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    indx_2_module_map: Dict[int, nn.Module],
) -> None:
    """Set destination layer without pooling for existing entry in `dynapcnn_layer_info`.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    edge: Tuple of 2 integers, indicating edge between two nodes in graph.
        Edge source has to be a neuron layer within an existing entry of
        `dynapcnn_layer_info`. Edge target has to be the weight layer of
        another dynapcnn layer.
    node_2_layer_map (dict): Maps each node ID to the ID of the layer it is assigned to.
        Will be updated in-place.
    nodes_io_shapes (dict): Map from node ID to dict containing node's in- and output shapes
    indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`
    """
    # Make sure both source (neuron layer) and target (weight layer) have been previously processed
    try:
        source_layer_idx = node_2_layer_map[edge[0]]
    except KeyError:
        neuron_layer = indx_2_module_map[edge[0]]
        raise InvalidGraphStructure(
            f"Spiking layer {neuron_layer} cannot be assigned to a dynapcnn layer. "
            "This is likely due to an unsupported SNN architecture. Spiking "
            "layers have to be preceded by a weight layer (`nn.Conv2d` or "
            "`nn.Linear`)."
        )
    try:
        destination_layer_idx = node_2_layer_map[edge[1]]
    except KeyError:
        weight_layer = indx_2_module_map[edge[1]]
        raise InvalidGraphStructure(
            f"Weight layer {weight_layer} cannot be assigned to a dynapcnn layer. "
            "This is likely due to an unsupported SNN architecture. Weight "
            "layers have to be followed by a spiking layer (`IAFSqueeze`)."
        )

    # Add new destination
    output_shape = nodes_io_shapes[edge[0]]["output"]
    layer_info = dynapcnn_layer_info[source_layer_idx]
    layer_info["destinations"].append(
        {
            "pooling_ids": [],
            "pooling_modules": [],
            "destination_layer": destination_layer_idx,
            "output_shape": output_shape,
        }
    )


def set_pooling_layer_destination(
    dynapcnn_layer_info: Dict[int, Dict],
    edge: Edge,
    node_2_layer_map: Dict[int, int],
    nodes_io_shapes: Dict[int, Dict[str, Tuple[Size, Size]]],
    indx_2_module_map: Dict[int, nn.Module],
) -> None:
    """Set destination layer with pooling for existing entry in `dynapcnn_layer_info`.

    Parameters
    ----------
    dynapcnn_layer_info: Dict with one entry for each future dynapcnn layer.
        key is unique dynapcnn layer ID, value is dict with nodes of the layer
        Will be updated in-place.
    edge: Tuple of 2 integers, indicating edge between two nodes in graph.
        Edge source has to be a pooling layer that is at the end of at least
        one pooling chain within an existing entry of `dynapcnn_layer_info`.
        Edge target has to be a weight layer within an existing entry of
        `dynapcnn_layer_info`.
    node_2_layer_map (dict): Maps each node ID to the ID of the layer it is assigned to.
        Will be updated in-place.
    nodes_io_shapes (dict): Map from node ID to dict containing node's in- and output shapes
    indx_2_module_map (dict): Maps node IDs of the graph as `key` to their associated module as `value`
    """
    # Make sure both source (pooling layer) and target (weight layer) have been previously processed
    try:
        source_layer_idx = node_2_layer_map[edge[0]]
    except KeyError:
        poolin_layer = indx_2_module_map[edge[0]]
        raise InvalidGraphStructure(
            f"Layer {poolin_layer} cannot be assigned to a dynapcnn layer. "
            "This is likely due to an unsupported SNN architecture. Pooling "
            "layers have to be preceded by a spiking layer (`IAFSqueeze`), "
            "another pooling layer, or DVS input"
        )
    try:
        destination_layer_idx = node_2_layer_map[edge[1]]
    except KeyError:
        weight_layer = indx_2_module_map[edge[1]]
        raise InvalidGraphStructure(
            f"Weight layer {weight_layer} cannot be assigned to a dynapcnn layer. "
            "This is likely due to an unsupported SNN architecture. Weight "
            "layers have to be preceded by a spiking layer (`IAFSqueeze`), "
            "another pooling layer, or DVS input"
        )

    # Find current source node within destinations
    layer_info = dynapcnn_layer_info[source_layer_idx]
    matched = False
    for destination in layer_info["destinations"]:
        if destination["pooling_ids"][-1] == edge[0]:
            if destination["destination_layer"] is not None:
                # Destination is already linked to a postsynaptic layer. This happens when
                # pooling nodes have outgoing edges to different weight layer.
                # Copy the destination
                # TODO: Add unit test for this case
                destination = {k: v for k, v in destination.items()}
                layer_info["destinations"].append(destination)
            matched = True
            break
    if not matched:
        poolin_layer = indx_2_module_map[edge[0]]
        raise InvalidGraphStructure(
            f"Layer {poolin_layer} cannot be assigned to a dynapcnn layer. "
            "This is likely due to an unsupported SNN architecture. Pooling "
            "layers have to be preceded by a spiking layer (`IAFSqueeze`), "
            "another pooling layer, or DVS input"
        )

    # Set destination layer within destination dict that holds current source node
    destination["destination_layer"] = destination_layer_idx
    output_shape = nodes_io_shapes[edge[0]]["output"]
    destination["output_shape"] = output_shape


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


def verify_layer_info(
    dynapcnn_layer_info: Dict[int, Dict], edge_counts: Optional[Dict[str, int]] = None
):
    """Verify that `dynapcnn_layer_info` matches formal requirements.

    - Every layer needs to have at least a `conv`, `neuron`, and `destinations`
        entry.
    - If `edge_counts` is provided, also make sure that number of layer matches
        numbers of edges.

    Parameters
    ----------
    - dynapcnn_layer_info: Dict with information to construct and connect
        DynapcnnLayer instances
    - edge_counts: Optional Dict with edge counts for each edge type. If not
        `None`, will be used to do further verifications on `dynapcnn_layer_info`

    Raises
    ------
    - InvalidGraphStructure: if any verification fails.
    """

    # Make sure that each dynapcnn layer has at least a weight layer and a neuron layer
    for idx, info in dynapcnn_layer_info.items():
        if not "conv" in info:
            raise InvalidGraphStructure(
                f"DynapCNN layer {idx} has no weight assigned, which should "
                "never happen. " + default_invalid_structure_string
            )
        if not "neuron" in info:
            raise InvalidGraphStructure(
                f"DynapCNN layer {idx} has no spiking layer assigned, which "
                "should never happen. " + default_invalid_structure_string
            )
        if not "destinations" in info:
            raise InvalidGraphStructure(
                f"DynapCNN layer {idx} has no destination info assigned, which "
                "should never happen. " + default_invalid_structure_string
            )
    if edge_counts is not None:
        # Make sure there are as many layers as edges from weight to neuron
        if edge_counts["weight-neuron"] - len(dynapcnn_layer_info) > 0:
            raise InvalidGraphStructure(
                "Not all weight-to-neuron edges have been processed, which "
                "should never happen. " + default_invalid_structure_string
            )
