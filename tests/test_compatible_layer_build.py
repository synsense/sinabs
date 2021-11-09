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
    layers = [sl.SumPool2d(2), nn.AvgPool2d(3), sl.IAF()]

    from sinabs.backend.dynapcnn.utils import construct_next_pooling_layer

    pool_lyr, layer_idx_next, rescale_factor = construct_next_pooling_layer(layers, 0)

    assert pool_lyr.kernel_size == (6,6)
    assert layer_idx_next == 2
    assert rescale_factor == 9


def test_non_square_pooling_kernel():
    layers = [
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.IAF(),
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
        sl.IAF(),
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
        sl.IAF(),
        nn.Conv2d(8, 2, kernel_size=3, stride=1, bias=False),
        sl.IAF(),
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
        sl.IAF(),
        sl.SumPool2d(2),
        nn.AvgPool2d(2),
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.IAF(),
        nn.Conv2d(2, 8, kernel_size=3, stride=1, bias=False),
        sl.IAF(),
    ]

    from sinabs.backend.dynapcnn.utils import construct_next_dynapcnn_layer

    dynapcnn_lyr, layer_idx_next, rescale_factor = construct_next_dynapcnn_layer(
        layers, 0, in_shape=(2, 28, 28), discretize=True, rescale_factor=1
    )

    print(dynapcnn_lyr)
    assert dynapcnn_lyr.pool_layer.kernel_size == (4, 4)
    assert layer_idx_next == 4
    assert rescale_factor == 4


def test_build_from_list_dynapcnn_layers_only():
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
        sl.IAF()
    ]

    from sinabs.backend.dynapcnn.utils import build_from_list

    chip_model = build_from_list(
        layers, in_shape=in_shape, discretize=True
    )

    assert len(chip_model) == 4
    assert chip_model[0].get_output_shape() == (8, 6, 6)
    assert chip_model[1].get_output_shape() == (16, 4, 4)
    assert chip_model[2].get_output_shape() == (2, 2, 2)
    assert chip_model[3].get_output_shape() == (5, 1, 1)


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
        build_from_list(
            layers, in_shape=in_shape, discretize=True
        )


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

