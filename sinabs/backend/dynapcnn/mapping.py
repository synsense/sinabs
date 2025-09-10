from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import sinabs

from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer
from .exceptions import InvalidModel


@dataclass
class LayerConstraints:
    kernel_memory: int
    neuron_memory: int
    bias_memory: int

    def fits(self, layer: DynapcnnLayer) -> bool:
        layer_summary = layer.memory_summary()
        return (
            (0 <= layer_summary["kernel"] <= self.kernel_memory)
            and (0 <= layer_summary["neuron"] <= self.neuron_memory)
            and (0 <= layer_summary["bias"] <= self.bias_memory)
        )


def find_chip_layers(
    layer: DynapcnnLayer, constraints: List[LayerConstraints]
) -> List[int]:
    """Find all layers where a given layer configuration fits.

    Args:
        layer: DynapcnnLayer.
        constraints: A list of all the layer's constraints.

    Returns:
        A list of indices of layers where the given layer fits.
    """
    idx = [i for (i, constraint) in enumerate(constraints) if constraint.fits(layer)]
    return idx


def get_valid_mapping(
    layers: Dict[int, DynapcnnLayer], constraints: List[LayerConstraints]
) -> Dict[int, int]:
    """Given a model, find a valid layer ordering for its placement within the constraints
    provided.

    Args:
        model: an instance of a DynapcnnNetwork or a DynapcnnNetworkGraph.
        constraints: a list of all the layer's constraints.

    Returns:
        Dict mapping from layer index (key) to assigned core ID (value).
    -------
    """
    # Store layer indices and lists of possible target chips in separate lists
    layer_indices = []
    layer_mapping = []
    for layer_index, this_layer in layers.items():
        # Skip DVSLayers
        if isinstance(this_layer, DynapcnnLayer):
            chip_layers = find_chip_layers(this_layer, constraints)
            layer_mapping.append(chip_layers)
            layer_indices.append(layer_index)
        # Make sure only DynapcnnLayers and DVSLayers are passed
        elif not isinstance(this_layer, DVSLayer):
            raise ValueError(f"Found unexpected layer type: `{type(this_layer)}")

    graph = make_flow_graph(layer_mapping, len(constraints))

    # use Edmonds' Algorithm to find suitable cores for each DynapcnnLayer.
    new_graph = edmonds(graph, 0, len(graph) - 1)
    netmap = recover_mapping(new_graph, len(layer_mapping))

    # Convert `netmap` to dict mapping from layer index to core ID
    return {layer_idx: core_id for layer_idx, core_id in zip(layer_indices, netmap)}


@dataclass
class FlowGraphEdge:
    s: int
    t: int
    cap: int
    flow: int = 0
    rev: Optional["FlowGraphEdge"] = None

    def __repr__(self):
        return f"FlowGraphEdge from {self.s} to {self.t} with capacity {self.cap} and flow {self.flow}"


def edmonds(
    graph: List[List[FlowGraphEdge]], source: int, sink: int, verbose: bool = False
) -> List[List[FlowGraphEdge]]:
    """Use Edmonds' Algorithm to compute flow of flow graph

    Makes a copy of the graph. The original graph is not changed in place.

    Args:
        graph List[List[FlowGraphEdge]]): Flow graph representation. Each list
            entry corresponds to a node and consists of a list holding the
            outgoing edges from this node.
        source (int): Index of source node within graph.
        sind (int): Index of sink node within graph.
        verbose (bool): Print detailed flow information if `True`.

    Returns:
        New flow graph with calculated flow. Type is List[List[FlowGraphEdge]].
    """
    graph = deepcopy(graph)
    flow = 0
    while True:
        q = deque()
        q.append(source)
        pred = [None for _ in range(len(graph))]
        while len(q) != 0:
            cur = q.popleft()  # current node index
            for edge in graph[cur]:  # edges to/from current node
                if pred[edge.t] is None and edge.t != source and edge.cap > edge.flow:
                    pred[edge.t] = edge
                    q.append(edge.t)
        if pred[sink] is not None:
            delta_flow = float("inf")
            edge = pred[sink]
            while edge is not None:
                prev_flow = delta_flow
                delta_flow = min(delta_flow, edge.cap - edge.flow)
                if (delta_flow < prev_flow) and verbose:
                    print(f"found new min flow {delta_flow} on edge {edge}")
                edge = pred[edge.s]
            edge = pred[sink]
            while edge is not None:
                edge.flow += delta_flow
                edge.rev.flow -= delta_flow
                edge = pred[edge.s]
            flow += delta_flow
        if pred[sink] is None:
            break
    return graph


def make_flow_graph(
    layer_mapping: List[List[int]], num_layers: int = 9
) -> List[List[FlowGraphEdge]]:
    """
    Make a bipartite flow graph (flow network) given all possible chip layers
    for each DynapcnnLayer layer. The goal is to formulate the mapping from
    software layer to chip layer as a bipartite matching problem. Note that the
    flows are not computed yet. The flow for the graph generated here needs to
    be populated by calling the method `edmonds`

    Args:
        layer_mapping: List of a list of matching chip core indices for each software layer.
            Eg. [[1,3], [4, 6, 1]] for a two layer model
        num_layers (int): Number of layers on the chip

    Returns:
        Flow graph representation. Each list entry corresponds to a node and consists
        of a list holding the outgoing edges from this node.
        The returned object is of type List[List[FlowGraphEdge]].
    """
    graph = []
    # add all our nodes
    # one source node
    graph.append([])
    # one node for every layer that will be mapped
    for __ in range(len(layer_mapping)):
        graph.append([])
    # one node for every chip layer
    for __ in range(num_layers):
        graph.append([])
    # one sink node
    graph.append([])
    # add all node edges
    target_offset = len(layer_mapping) + 1
    # first from source to all layers
    for i in range(len(layer_mapping)):
        source_to_layer = FlowGraphEdge(s=0, t=i + 1, cap=1, flow=0)
        layer_to_source = FlowGraphEdge(s=i + 1, t=0, cap=0, flow=0)
        # fill in reverse pointers
        source_to_layer.rev = layer_to_source
        layer_to_source.rev = source_to_layer
        # append new edges
        graph[0].append(source_to_layer)
        graph[i + 1].append(layer_to_source)
    # then from layers to chip layers
    for i, layer_targets in enumerate(layer_mapping):
        for target in layer_targets:
            layer_to_chip = FlowGraphEdge(
                s=i + 1, t=target + target_offset, cap=1, flow=0
            )
            chip_to_layer = FlowGraphEdge(
                s=target + target_offset, t=i + 1, cap=0, flow=0
            )
            layer_to_chip.rev = chip_to_layer
            chip_to_layer.rev = layer_to_chip
            graph[i + 1].append(layer_to_chip)
            graph[target + target_offset].append(chip_to_layer)
    # then from chip layers to sink
    sink = len(graph) - 1
    for chip_node in range(target_offset, sink):
        graph[chip_node].append(FlowGraphEdge(s=chip_node, t=sink, cap=1, flow=0))
        graph[sink].append(FlowGraphEdge(s=sink, t=chip_node, cap=0, flow=0))
        graph[chip_node][-1].rev = graph[sink][-1]
        graph[sink][-1].rev = graph[sink][-1]
    return graph


def recover_mapping(graph: List[List[FlowGraphEdge]], num_layers: int) -> List[int]:
    """Based on the flow graph retrieve a layer-to-core mapping

    Args:
        graph List[List[FlowGraphEdge]]): Flow graph representation with flow
            calculated. Each list entry corresponds to a node and consists of a
            list holding the outgoing edges from this node.
        num_layers (int): Number of software layers.

    Returns:
        Assigned core IDs for each layer in order. Type is List[int].
    """
    mapping = []
    for i in range(1, num_layers + 1):  # `+1` to skip source node
        for edge in graph[i]:
            if edge.flow == 1:
                mapping.append(edge.t - num_layers - 1)
    if len(mapping) != num_layers:
        # TODO - check if this error message make sense with nonseq implementation
        raise ValueError(
            "No valid mapping found. "
            "For Speck family you can use `utils.validate_memory_mapping_speck()` to get more information."
        )
    return mapping
