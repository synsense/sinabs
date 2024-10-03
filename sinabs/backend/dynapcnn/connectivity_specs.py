"""
functionality : list device-independent supported connections between layers on chip
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

from typing import Union

import torch.nn as nn

import sinabs.layers as sl

Pooling = Union[sl.SumPool2d, nn.AvgPool2d]
Weight = Union[nn.Conv2d, nn.Linear]
Neuron = sl.IAFSqueeze

VALID_SINABS_EDGES = {
    # convoluion is always followed by a neuron layer.
    0: (Weight, Neuron),
    # Neuron layer can be followed by pooling
    1: (Neuron, Pooling),
    # Pooling can be followed by another pooling (will be consolidated)
    2: (Pooling, Pooling),
    # Neuron layer can be followed by weight layer of next core
    3: (Neuron, Weight),
    # Pooling can be followed by weight layer of next core
    4: (Pooling, Weight),
}
VALID_SINABS_EDGE_TYPE_IDS = {v: k for k, v in VALID_SINABS_EDGES.items()}

# Between two cores only neuron->weight or pooling->weight connections are possible
VALID_DYNAPCNNLAYER_EDGES = [(Neuron, Weight), (Pooling, Weight)]

# Only `Merge` layers are allowed to join multiple inputs
LAYER_TYPES_WITH_MULTIPLE_INPUTS = Union[sl.Merge]

# Neuron and pooling layers can have their output sent to multiple cores
LAYER_TYPES_WITH_MULTIPLE_OUTPUTS = Union[Neuron, Pooling]

