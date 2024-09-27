"""
functionality : implementation of exceptions and constraints regarding the processing of edges from a network
                computational graph.
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

from typing import Tuple

import torch.nn as nn

import sinabs.layers as sl

# Constraints.                                   # @TODO constraints are ideally device-dependent.

VALID_SINABS_EDGES = {
    0: (
        nn.Conv2d,
        sl.iaf.IAFSqueeze,
    ),  # convoluion is always followed by a neuron layer.
    1: (sl.iaf.IAFSqueeze, nn.AvgPool2d),
    2: (sl.iaf.IAFSqueeze, nn.Conv2d),
    3: (
        sl.iaf.IAFSqueeze,
        nn.Linear,
    ),  # same case as `2` since `nn.Linear` layers are converted into `nn.Conv2d` by `DynapcnnLayer`.
    4: (
        nn.AvgPool2d,
        nn.Conv2d,
    ),  # `nn.Pool2d` is always "ending" a DynapcnnLayer sequence of modules (comes after a `sl.iaf`).
    5: (
        nn.AvgPool2d,
        nn.Linear,
    ),  # same as case `4` since `nn.Linear` layers are converted into `nn.Conv2d` by `DynapcnnLayer`.
    6: (
        nn.Linear,
        sl.iaf.IAFSqueeze,
    ),  # same as case `0` since `nn.Linear` layers are converted into `nn.Conv2d` by `DynapcnnLayer`.
    7: (
        sl.iaf.IAFSqueeze,
        sl.SumPool2d,
    ),  # same as key `1` but with `sl.SumPool2d` instead.
    8: (sl.SumPool2d, nn.Conv2d),  # same as key `4` but with `sl.SumPool2d` instead.
    9: (sl.SumPool2d, nn.Linear),  # same as key `5` but with `sl.SumPool2d` instead.
}
VALID_SINABS_EDGE_TYPE_IDS = {v: k for k, v in VALID_SINABS_EDGES}

VALID_DYNAPCNNLAYER_EDGES = [
    (sl.iaf.IAFSqueeze, nn.Conv2d),
    (
        sl.iaf.IAFSqueeze,
        nn.Linear,
    ),  # `nn.Linear` layers are converted into `nn.Conv2d` by `DynapcnnLayer`.
    (nn.AvgPool2d, nn.Conv2d),
    (sl.SumPool2d, nn.Conv2d),
    (
        nn.AvgPool2d,
        nn.Linear,
    ),  # `nn.Linear` layers are converted into `nn.Conv2d` by `DynapcnnLayer`.
    (sl.SumPool2d, nn.Linear),
]

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
        super().__init__(
            f"Node {node} in edge {edge} can not found in previously processed edges."
        )


class UnknownNode(Exception):
    node: int

    def __init__(self, node):
        super().__init__(
            f"Node {node} can not be found within any DynapcnnLayer mapper."
        )


class MaxDestinationsReached(Exception):
    dynapcnnlayer_index: int

    def __init__(self, dynapcnnlayer_index):
        super().__init__(
            f"DynapcnnLayer with index {dynapcnnlayer_index} has more than 2 destinations."
        )


class InvalidLayerLoop(Exception):
    dynapcnnlayerA_index: int
    dynapcnnlayerB_index: int

    def __init__(self, dynapcnnlayerA_index, dynapcnnlayerB_index):
        super().__init__(
            f"DynapcnnLayer {dynapcnnlayerA_index} can not connect to {dynapcnnlayerB_index} since reverse edge already exists."
        )


class InvalidLayerDestination(Exception):
    dynapcnnlayerA: type
    dynapcnnlayerB: type

    def __init__(self, dynapcnnlayerA, dynapcnnlayerB):
        super().__init__(
            f"DynapcnnLayer {dynapcnnlayerA} in one core can not connect to {dynapcnnlayerB} in another core."
        )
