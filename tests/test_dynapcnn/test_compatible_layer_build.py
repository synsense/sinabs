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
            (2, nn.AvgPool2d, [2, 2], 1./4),
            ((2, 2), sl.nn.AvgPool2d, [2, 2], 1./4),
            (3, sl.nn.AvgPool2d, [3, 3], 1./9),
            ((4, 4), sl.nn.AvgPool2d, [4, 4], 1./16),
        ]
)
def test_construct_pooling_from_1_layer(pooling, layer_type, expected_pooling, expected_scaling):
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
    assert scaling == 1./9


# TODO: Move these fail cases to another test. Rename this file or move other tests as well.
def test_missing_spiking_layer():
    in_shape = (2, 28, 28)
    layers = [
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.IAF(),
        sl.SumPool2d(2),
        nn.AvgPool2d(2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, bias=False),
        sl.IAF(),
        nn.Dropout2d(),
        nn.Conv2d(16, 2, kernel_size=3, stride=1, bias=False),
        sl.IAF(),
        nn.Flatten(),
        nn.Linear(8, 5),
    ]
    from sinabs.backend.dynapcnn.exceptions import MissingLayer
    from sinabs.backend.dynapcnn.utils import build_from_list

    with pytest.raises(MissingLayer):
        build_from_list(layers, in_shape=in_shape, discretize=True)


def test_incorrect_model_start():
    in_shape = (2, 28, 28)
    layers = [
        sl.IAF(),
        sl.SumPool2d(2),
        nn.AvgPool2d(2),
    ]
    from sinabs.backend.dynapcnn.exceptions import UnexpectedLayer
    from sinabs.backend.dynapcnn.utils import construct_next_dynapcnn_layer

    with pytest.raises(UnexpectedLayer):
        construct_next_dynapcnn_layer(
            layers, 0, in_shape=in_shape, discretize=True, rescale_factor=1
        )