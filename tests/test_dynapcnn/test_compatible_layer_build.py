import pytest
import torch.nn as nn

import sinabs.layers as sl


@pytest.mark.parametrize(
    ("pooling", "layer_type", "expected_pooling", "expected_scaling"),
    [
        (2, sl.SumPool2d, [2, 2], 1),
        ((2, 2), sl.SumPool2d, [2, 2], 1),
        (3, sl.SumPool2d, [3, 3], 1),
        ((4, 4), sl.SumPool2d, [4, 4], 1),
        (2, nn.AvgPool2d, [2, 2], 1.0 / 4),
        ((2, 2), nn.AvgPool2d, [2, 2], 1.0 / 4),
        (3, nn.AvgPool2d, [3, 3], 1.0 / 9),
        ((4, 4), nn.AvgPool2d, [4, 4], 1.0 / 16),
    ],
)
def test_construct_pooling_from_1_layer(
    pooling, layer_type, expected_pooling, expected_scaling
):
    layers = [layer_type(pooling)]

    from sinabs.backend.dynapcnn.dynapcnn_layer_utils import consolidate_dest_pooling

    cumulative_pooling, scaling = consolidate_dest_pooling(layers)

    assert cumulative_pooling == expected_pooling
    assert scaling == expected_scaling


def test_construct_pooling_from_2_layers():
    layers = [sl.SumPool2d(2), nn.AvgPool2d(3)]

    from sinabs.backend.dynapcnn.dynapcnn_layer_utils import consolidate_dest_pooling

    cumulative_pooling, scaling = consolidate_dest_pooling(layers)

    assert cumulative_pooling == [6, 6]
    assert scaling == 1.0 / 9
