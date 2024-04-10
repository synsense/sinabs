from typing import Tuple
import torch.nn as nn
import sinabs.layers as sl

VALID_SINABS_EDGES = [
    (nn.Conv2d, sl.iaf.IAFSqueeze),
    (sl.iaf.IAFSqueeze, nn.AvgPool2d),
    (sl.iaf.IAFSqueeze, nn.Conv2d),
    (sl.iaf.IAFSqueeze, nn.Linear),
    (nn.AvgPool2d, nn.Conv2d),
    (nn.AvgPool2d, nn.Linear),
    (nn.Linear, sl.iaf.IAFSqueeze),
]

VALID_SINABS_NODE_FAN_IN = []
VALID_SINABS_NODE_FAN_OUT = []

def process_edge(layers: list, edge: Tuple[int, int], mapper: dict):
    """ ."""
    edge_layers = (type(layers[edge[0]]), type(layers[edge[1]]))

    if edge_layers in VALID_SINABS_EDGES:
        print(edge)
    else:
        ...