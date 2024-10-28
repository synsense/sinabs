"""
functionality : list device-independent supported connections between layers on chip
"""

from typing import Union

import torch.nn as nn

import sinabs.layers as sl
from dvs_layer import DVSLayer

Pooling = (sl.SumPool2d, nn.AvgPool2d)
Weight = (nn.Conv2d, nn.Linear)
Neuron = (sl.IAFSqueeze,)
Dvs = (DVSLayer,)

# @TODO - need to list other edge cases involving DVS layer (for now only dvs-weight and dvs-pooling).
VALID_SINABS_EDGE_TYPES_ABSTRACT = {
    # convoluion is always followed by a neuron layer.
    (Weight, Neuron): "weight-neuron",
    # Neuron layer can be followed by pooling
    (Neuron, Pooling): "neuron-pooling",
    # Pooling can be followed by another pooling (will be consolidated)
    (Pooling, Pooling): "pooling-pooling",
    # Neuron layer can be followed by weight layer of next core
    (Neuron, Weight): "neuron-weight",
    # Pooling can be followed by weight layer of next core
    (Pooling, Weight): "pooling-weight",
    # Dvs can be followed by weight layer of next core
    (Dvs, Weight): "dvs-weight",
    # Dvs can be followed by pooling layers
    (Dvs, Pooling): "dvs-pooling",
}

# Unpack dict
VALID_SINABS_EDGE_TYPES = {
    (source_type, target_type): name
    for types, name in VALID_SINABS_EDGE_TYPES_ABSTRACT.items()
    for source_type in types[0]
    for target_type in types[1]
}

# Only `Merge` layers are allowed to join multiple inputs
LAYER_TYPES_WITH_MULTIPLE_INPUTS = Union[sl.Merge]

# Neuron and pooling layers can have their output sent to multiple cores
LAYER_TYPES_WITH_MULTIPLE_OUTPUTS = Union[(*Neuron, *Pooling, *Dvs)]
