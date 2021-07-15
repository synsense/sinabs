import pytest
import torch.nn as nn
import sinabs.layers as sl


def test_construct_pooling_from_1_layer():
    layers = [sl.SumPool2d(2)]

    from sinabs.backend.dynapcnn.utils import construct_next_pooling_layer

    pool_lyr, layer_idx_next, rescale_factor = construct_next_pooling_layer(layers, 0)

    assert pool_lyr.kernel_size == (2,2)
    assert layer_idx_next == 1
    assert rescale_factor == 1


def test_construct_pooling_from_2_layers():
    layers = [sl.SumPool2d(2), nn.AvgPool2d(3), sl.SpikingLayer()]

    from sinabs.backend.dynapcnn.utils import construct_next_pooling_layer

    pool_lyr, layer_idx_next, rescale_factor = construct_next_pooling_layer(layers, 0)

    assert pool_lyr.kernel_size == (6,6)
    assert layer_idx_next == 2
    assert rescale_factor == 9


def test_non_square_pooling_kernel():
    layers = [
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.SpikingLayer(),
        sl.SumPool2d((2, 3))
    ]

    from sinabs.backend.dynapcnn.utils import construct_next_dynapcnn_layer

    with pytest.raises(ValueError):
        _ = construct_next_dynapcnn_layer(
            layers, 0, in_shape=(2, 28, 28), discretize=True, rescale_factor=1
        )


def test_construct_dynapcnn_layer_from_3_layers():
    layers = [
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.SpikingLayer(),
        sl.SumPool2d(2)
    ]

    from sinabs.backend.dynapcnn.utils import construct_next_dynapcnn_layer

    dynapcnn_lyr, layer_idx_next, rescale_factor = construct_next_dynapcnn_layer(
        layers, 0, in_shape=(2, 28, 28), discretize=True, rescale_factor=1
    )

    print(dynapcnn_lyr)
    assert layer_idx_next == 3
    assert rescale_factor == 1


def test_construct_dynapcnn_layer_no_pool_layers():
    layers = [
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.SpikingLayer(),
        nn.Conv2d(8, 2, kernel_size=3, stride=1, bias=False),
        sl.SpikingLayer(),
    ]

    from sinabs.backend.dynapcnn.utils import construct_next_dynapcnn_layer

    dynapcnn_lyr, layer_idx_next, rescale_factor = construct_next_dynapcnn_layer(
        layers, 0, in_shape=(2, 28, 28), discretize=True, rescale_factor=1
    )

    print(dynapcnn_lyr)
    assert layer_idx_next == 2
    assert rescale_factor == 1


def test_construct_dynapcnn_layer_from_8_layers():
    layers = [
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.SpikingLayer(),
        sl.SumPool2d(2),
        nn.AvgPool2d(2),
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.SpikingLayer(),
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.SpikingLayer(),
    ]

    from sinabs.backend.dynapcnn.utils import construct_next_dynapcnn_layer

    dynapcnn_lyr, layer_idx_next, rescale_factor = construct_next_dynapcnn_layer(
        layers, 0, in_shape=(2, 28, 28), discretize=True, rescale_factor=1
    )

    print(dynapcnn_lyr)
    assert dynapcnn_lyr.pool_layer.kernel_size == (4, 4)
    assert layer_idx_next == 4
    assert rescale_factor == 4
