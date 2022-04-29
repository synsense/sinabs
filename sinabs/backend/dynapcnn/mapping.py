from typing import List, Tuple, Optional
from dataclasses import dataclass
from copy import deepcopy
from collections import deque
from .dvs_layer import DVSLayer
from .dynapcnn_layer import DynapcnnLayer


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


def find_chip_layers(layer: DynapcnnLayer, constraints: List[LayerConstraints]) -> List[int]:
    """
    Find all layers where a given layer configuration fits.

    Parameters
    ----------
    layer:
        DynapcnnLayer

    constraints:
        A list of all the layer's constraints

    Returns
    -------
        A list of indices of layers where the given layer fits.
    """
    idx = [i for (i, constraint) in enumerate(constraints) if constraint.fits(layer)]
    return idx


def get_valid_mapping(model: "DynapcnnNetwork", constraints: List[LayerConstraints]) -> List[Tuple[int, int]]:
    """
    Given a model, find a valid layer ordering for its placement within the constraints provided.

    Parameters
    ----------
    model:
        DynapcnnNetwork
    constraints:
        A list of all the layer's constraints

    Returns
    -------

    """
    layer_mapping = []

    for layer in model.compatible_layers:
        if isinstance(layer, DynapcnnLayer):
            layer_mapping.append(find_chip_layers(layer, constraints))

    graph = make_flow_graph(layer_mapping, len(constraints))

    # Call mapping
    new_graph = edmonds(graph, 0, len(graph) - 1)

    netmap = recover_mapping(new_graph, layer_mapping)
    return netmap


@dataclass
class Edge:
    s: int
    t: int
    cap: int
    flow: int = 0
    rev: Optional["Edge"] = None

    def __repr__(self):
        return f"Edge from {self.s} to {self.t} with capacity {self.cap} and flow {self.flow}"


# graph is list of list of edges. Each edge is
def edmonds(graph, source, sink, verbose: bool = False):
    graph = deepcopy(graph)
    flow = 0
    while True:
        q = deque()
        q.append(source)
        pred = [None for _ in range(len(graph))]
        while len(q) != 0:
            cur = q.popleft()
            for edge in graph[cur]:
                if pred[edge.t] is None and edge.t != source and edge.cap > edge.flow:
                    pred[edge.t] = edge
                    q.append(edge.t)
        if pred[sink] is not None:
            delta_flow = float('inf')
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


def make_flow_graph(layer_mapping: List[List[int]], num_layers: int = 9) -> List[List[Edge]]:
    """
    Make a flow graph given all possible chip layers for each DynapcnnCompatibleLayer layer.
    Note that the flows are not computed yet.
    The flow for the graph generated here needs to be populated by calling the method `edmonds`

    Parameters
    ----------
    layer_mapping:
        List of a list of all layer indices. Eg. [[1,3], [4, 6, 1]] for a two layer model
    num_layers:
        Number of layers on the chip

    Returns
    -------
        graph: List[List[Edge]]
    """
    graph = []
    # add all our nodes
    # one source node
    graph.append([])
    # one node for every layer that will be mapped
    for x in range(len(layer_mapping)):
        graph.append([])
    # one node for every chip layer
    for x in range(num_layers):
        graph.append([])
    # one sink node
    graph.append([])
    # add all node edges
    target_offset = len(layer_mapping) + 1
    # first from source to all layers
    for i in range(len(layer_mapping)):
        graph[0].append(Edge(s=0, t=i + 1, cap=1, flow=0))
        # add the reverse edge
        graph[i + 1].append(Edge(s=i + 1, t=0, cap=0, flow=0))
        # fill in reverse pointers
        graph[0][-1].rev = graph[i + 1][-1]
        graph[i + 1][-1].rev = graph[0][-1]
    # then from layers to chip layers
    for i, layer_targets in enumerate(layer_mapping):
        for target in layer_targets:
            graph[i + 1].append(Edge(s=i + 1, t=target + target_offset, cap=1, flow=0))
            graph[target + target_offset].append(Edge(s=target + target_offset, t=i + 1, cap=0, flow=0))
            graph[i + 1][-1].rev = graph[target + target_offset][-1]
            graph[target + target_offset][-1].rev = graph[i + 1][-1]
    # print(graph)
    # then from chip layers to sink
    for i, layer in enumerate(graph[target_offset:-1]):
        sink = len(graph) - 1
        source = i + target_offset
        graph[source].append(Edge(s=source, t=sink, cap=1, flow=0))
        graph[sink].append(Edge(s=sink, t=source, cap=0, flow=0))
        graph[source][-1].rev = graph[sink][-1]
        graph[sink][-1].rev = graph[sink][-1]
    return graph


def recover_mapping(graph, layer_mapping) -> List[Tuple[int, int]]:
    mapping = []
    for i, layer in enumerate(layer_mapping):
        for edge in graph[i + 1]:
            if edge.flow == 1:
                mapping.append((i, edge.t - len(layer_mapping) - 1))
    if len(mapping) != len(layer_mapping):
        raise ValueError("No valid mapping found")
    return mapping
#
#
### Chip specific constraints
#_WEIGHTS_MEMORY_SIZE = [
#    16 * 1024,  # 0
#    16 * 1024,  # 1
#    16 * 1024,  # 2
#    32 * 1024,  # 3
#    32 * 1024,  # 4
#    64 * 1024,  # 5
#    64 * 1024,  # 6
#    16 * 1024,  # 7
#    16 * 1024,
#]  # _WEIGHTS_MEMORY_SIZE
#
#_NEURONS_MEMORY_SIZE = [
#    64 * 1024,  # 0
#    64 * 1024,  # 1
#    64 * 1024,  # 2
#    32 * 1024,  # 3
#    32 * 1024,  # 4
#    16 * 1024,  # 5
#    16 * 1024,  # 6
#    16 * 1024,  # 7
#    16 * 1024,
#]  # 8
#_BIAS_MEMORY_SIZE = [1024] * 9
#
#dynapcnndevkit_constraints = [
#    LayerConstraints(km, nm, bm) for (km, nm, bm) in zip(_WEIGHTS_MEMORY_SIZE, _NEURONS_MEMORY_SIZE, _BIAS_MEMORY_SIZE)
#]
#
#speck2_constraints = dynapcnndevkit_constraints
#