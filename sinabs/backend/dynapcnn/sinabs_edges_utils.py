import sinabs.layers as sl
import torch.nn as nn
from typing import Tuple

# Constraints. @TODO this constraints are ideally device-dependent.

VALID_SINABS_EDGES = {
    0: (nn.Conv2d, sl.iaf.IAFSqueeze),                  # 'nn.Conv2d' is always followed by a 'sl.iaf'.
    1: (sl.iaf.IAFSqueeze, nn.AvgPool2d),
    2: (sl.iaf.IAFSqueeze, nn.Conv2d),
    3: (sl.iaf.IAFSqueeze, nn.Linear),                  # 'nn.Linear' layers are converted into 'nn.Conv2d' by 'DynapcnnLayer'.
    4: (nn.AvgPool2d, nn.Conv2d),                       # 'nn.Pool2d' is always "ending" a DynapcnnLayer sequence of modules (comes after a 'sl.iaf').
    5: (nn.AvgPool2d, nn.Linear),                       # 'nn.Linear' layers are converted into 'nn.Conv2d' by 'DynapcnnLayer'.
    6: (nn.Linear, sl.iaf.IAFSqueeze),
}

VALID_SINABS_NODE_FAN_IN = []
VALID_SINABS_NODE_FAN_OUT = []

# Edge exceptions.

class InvalidEdge(Exception):
    edge: Tuple[int, int]
    source: type
    target: type

    def __init__(self, edge, source, target):
        super().__init__(f"Invalid edge {edge}: {source} can not target {target}.")

class InvalidEdgeType(Exception):
    edge: Tuple[int, int]
    type: int

    def __init__(self, edge, type):
        super().__init__(f"Invalid edge type {type} for edge {edge}.")

class UnmatchedNode(Exception):
    edge: Tuple[int, int]
    node: int

    def __init__(self, edge, node):
        super().__init__(f"Node {node} in edge {edge} can not found in previously processed edges.")