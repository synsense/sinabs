"""
functionality : list device-independent supported connections between layers on chip
author        : Willian Soares Girao
contact       : williansoaresgirao@gmail.com
"""

from typing import Union

import torch.nn as nn

import sinabs.layers as sl

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
VALID_SINABS_EDGE_TYPE_IDS = {v: k for k, v in VALID_SINABS_EDGES.items()}

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

LAYER_TYPES_WITH_MULTIPLE_INPUTS = Union[sl.Merge]

LAYER_TYPES_WITH_MULTIPLE_OUTPUTS = Union[
    sl.IAFSqueeze, sl.SumPool2d, nn.AvgPool2d
]

