import torch
import torch.nn as nn


def test_init_defaults():
    from sinabs.backend.dynapcnn.dvs_layer import DVSLayer

    dvs_layer = DVSLayer(input_shape=(128, 128))

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = dvs_layer(data)

    assert (out == data).all()


def test_from_layers_empty():
    from sinabs.backend.dynapcnn.dvs_layer import DVSLayer
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.flipdims import FlipDims

    dvs_layer = DVSLayer.from_layers(input_shape=(2, 128, 128))

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = dvs_layer(data)

    assert (out == data).all()


def test_from_layers():
    from sinabs.backend.dynapcnn.dvs_layer import DVSLayer
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.crop2d import Crop2d

    pool_layer = sl.SumPool2d(2)
    crop_layer = Crop2d(((0, 59), (0, 54)))

    dvs_layer = DVSLayer.from_layers(input_shape=(2, 128, 128), pool_layer=pool_layer, crop_layer=crop_layer)

    print(dvs_layer)

    print(dvs_layer.get_config_dict())

    data = torch.rand((1, 2, 128, 128))

    out = crop_layer(data)
    assert out.shape == (1, 2, 59, 54)

    out = dvs_layer(data)

    assert out.shape == (1, 2, (64 - 5), (64 - 10))

    assert dvs_layer.get_roi() == ((0, 59), (0, 54))


def test_construct_empty():
    from sinabs.backend.dynapcnn.utils import construct_dvs_layer

    layers = []

    dvs_layer, layer_idx_next, rescale_factor = construct_dvs_layer(layers, input_shape=(2, 128, 128))

    assert rescale_factor == 1
    assert layer_idx_next == 0
    assert dvs_layer is None


def test_construct_from_sumpool():
    from sinabs.backend.dynapcnn.utils import construct_dvs_layer
    import sinabs.layers as sl

    layers = [sl.SumPool2d(2), sl.Cropping2dLayer(((1, 1), (1, 1)))]

    dvs_layer, layer_idx_next, rescale_factor = construct_dvs_layer(layers, input_shape=(2, 128, 128))

    print(dvs_layer)

    assert rescale_factor == 1
    assert layer_idx_next == 2
    assert dvs_layer.get_roi() == ((1, 63), (1, 63))


def test_convert_cropping2dlayer_to_crop2d():
    import sinabs.layers as sl
    from sinabs.backend.dynapcnn.utils import convert_cropping2dlayer_to_crop2d

    input_shape = (64, 50)
    cropping_lyr = sl.Cropping2dLayer(((1, 4), (3, 7)))
    crop2d_lyr = convert_cropping2dlayer_to_crop2d(cropping_lyr, input_shape)

    assert crop2d_lyr.top_crop == cropping_lyr.top_crop
    assert crop2d_lyr.left_crop == cropping_lyr.left_crop
    assert crop2d_lyr.bottom_crop == 64-4
    assert crop2d_lyr.right_crop == 50-7
